mod usage;

use anyhow::{Context as _, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, SinkExt, StreamExt, future::BoxFuture, future::Shared};
use gpui::{App, AsyncApp, Context, Entity, SharedString, Task, TaskExt as _, WeakEntity};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest, RequestBuilderExt as _,
    http::{HeaderName, HeaderValue},
};
use language_model::{
    CompactionResult, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelEffortLevel, LanguageModelId, LanguageModelName, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelRequest, LanguageModelToolChoice, RateLimiter,
};
use open_ai::{
    ReasoningEffort,
    responses::{ResponseInputItem, stream_response_ref},
};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{collections::BTreeMap, sync::Arc};
use url::form_urlencoded;
use util::ResultExt as _;

use open_ai::completion::{OpenAiResponseEventMapper, into_open_ai_response};

pub const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("openai-subscribed");
pub const PROVIDER_NAME: LanguageModelProviderName =
    LanguageModelProviderName::new("ChatGPT Subscription");

const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const OPENAI_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OPENAI_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

const CREDENTIALS_KEY: &str = "https://chatgpt.com/backend-api/codex";
const TOKEN_REFRESH_BUFFER_MS: u64 = Duration::from_mins(5).as_millis() as u64;
/// Requests the complete account catalog without Codex CLI version filtering.
///
/// The backend treats this exact version as an ungated sentinel. Other versions
/// are compared with each model's `minimal_client_version`.
const UNGATED_MODEL_CATALOG_CLIENT_VERSION: &str = "0.0.0";
// Codex applies the same bound because model discovery is a startup-critical request.
const MODEL_CATALOG_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CodexCredentials {
    access_token: String,
    refresh_token: String,
    expires_at_ms: u64,
    account_id: Option<String>,
    email: Option<String>,
    #[serde(default)]
    scopes: Vec<String>,
}

impl CodexCredentials {
    fn is_expired(&self) -> bool {
        let now = now_ms();
        now + TOKEN_REFRESH_BUFFER_MS >= self.expires_at_ms
    }
}

enum SignInState {
    Idle,
    Authorizing(Task<Result<()>>),
    PersistingCredentials { _task: Task<Result<()>> },
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccountSwitchPolicy {
    Manual,
    #[default]
    OnError,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct Account {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    credentials: Option<CodexCredentials>,
    #[serde(default)]
    needs_reauth: bool,
    #[serde(default)]
    billing_blocked: bool,
    #[serde(default)]
    unavailable_until_ms: u64,
    #[serde(default)]
    quota_until_ms: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    blocked_models: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    status: String,
}

impl Account {
    fn available(&self, model: &str) -> bool {
        self.credentials.is_some()
            && !self.needs_reauth
            && !self.billing_blocked
            && self.unavailable_until_ms <= now_ms()
            && self.quota_until_ms <= now_ms()
            && !self.blocked_models.iter().any(|blocked| blocked == model)
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct Accounts {
    accounts: BTreeMap<String, Account>,
    selected: String,
}

impl Accounts {
    fn insert(&mut self, credentials: CodexCredentials) {
        let id = credentials
            .account_id
            .clone()
            .or_else(|| credentials.email.clone())
            .unwrap_or_else(|| "legacy".into());
        let mut account = Account {
            credentials: Some(credentials),
            ..Default::default()
        };
        if let Some(previous) = self.accounts.get(&id) {
            account.quota_until_ms = previous.quota_until_ms;
            if account.quota_until_ms > now_ms() {
                account.status = "Plan quota reached".into();
            }
        }
        self.accounts.insert(id.clone(), account);
        self.selected = id;
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if let Ok(accounts) = serde_json::from_slice::<Self>(bytes) {
            return Ok(accounts);
        }
        let mut accounts = Self::default();
        accounts.insert(serde_json::from_slice(bytes)?);
        Ok(accounts)
    }
}

fn account_credentials_key(id: &str) -> String {
    let digest = Sha256::digest(id.as_bytes());
    format!(
        "{CREDENTIALS_KEY}/accounts/{}",
        URL_SAFE_NO_PAD.encode(digest)
    )
}

async fn load_accounts(provider: &dyn CredentialsProvider, cx: &AsyncApp) -> Result<Accounts> {
    let Some((_, bytes)) = provider.read_credentials(CREDENTIALS_KEY, cx).await? else {
        return Ok(Accounts::default());
    };
    let mut accounts = Accounts::decode(&bytes)?;
    for (id, account) in &mut accounts.accounts {
        if account.credentials.is_none() {
            let key = account_credentials_key(id);
            if let Some((_, bytes)) = provider.read_credentials(&key, cx).await? {
                account.credentials = Some(serde_json::from_slice(&bytes)?);
            } else {
                account.needs_reauth = true;
                account.status = "Saved credentials are missing. Sign in again".into();
            }
        }
    }
    Ok(accounts)
}

async fn save_accounts(
    provider: &dyn CredentialsProvider,
    accounts: &Accounts,
    cx: &AsyncApp,
) -> Result<()> {
    let previous = match provider.read_credentials(CREDENTIALS_KEY, cx).await? {
        Some((_, bytes)) => match Accounts::decode(&bytes) {
            Ok(previous) => previous,
            Err(error) if accounts.accounts.is_empty() => {
                log::warn!("Removing unreadable ChatGPT credential index: {error}");
                Accounts::default()
            }
            Err(error) => return Err(error),
        },
        None => Accounts::default(),
    };
    let mut index = accounts.clone();
    for (id, account) in &mut index.accounts {
        if let Some(credentials) = account.credentials.take() {
            let key = account_credentials_key(id);
            let bytes = serde_json::to_vec(&credentials)?;
            provider
                .write_credentials(&key, "Bearer", &bytes, cx)
                .await?;
        }
    }
    // Publish the index only after all credential records exist, preserving migration on failure.
    if accounts.accounts.is_empty() {
        provider.delete_credentials(CREDENTIALS_KEY, cx).await?;
    } else {
        let bytes = serde_json::to_vec(&index)?;
        provider
            .write_credentials(CREDENTIALS_KEY, "Accounts", &bytes, cx)
            .await?;
    }
    for id in previous
        .accounts
        .keys()
        .filter(|id| !accounts.accounts.contains_key(*id))
    {
        provider
            .delete_credentials(&account_credentials_key(id), cx)
            .await?;
    }
    Ok(())
}

pub struct State {
    accounts: Accounts,
    usage_requests: BTreeMap<String, usage::UsageRequest>,
    configured_policy: Option<AccountSwitchPolicy>,
    persistence_task: Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    sign_in_state: SignInState,
    refresh_tasks: BTreeMap<String, Shared<Task<Result<CodexCredentials, Arc<anyhow::Error>>>>>,
    load_task: Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    client_version: SharedString,
    available_models: Vec<ChatGptModel>,
    auth_generation: u64,
    model_catalog_generation: u64,
    last_auth_error: Option<SharedString>,
    last_model_catalog_error: Option<SharedString>,
}

#[derive(Debug)]
enum RefreshError {
    Fatal(anyhow::Error),
    Transient(anyhow::Error),
}

impl std::fmt::Display for RefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefreshError::Fatal(e) => write!(f, "{e}"),
            RefreshError::Transient(e) => write!(f, "{e}"),
        }
    }
}

impl State {
    /// Creates state and starts loading persisted credentials.
    ///
    /// Model discovery requests the ungated account catalog because host
    /// application versions are unrelated to Codex CLI compatibility versions.
    ///
    /// [`State::load_task`] resolves once the load finishes.
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Self>,
    ) -> Self {
        let load_task = cx
            .spawn({
                let credentials_provider = credentials_provider.clone();
                async move |this, cx| {
                    let result = load_accounts(credentials_provider.as_ref(), cx).await;
                    let refresh_models_task = this.update(cx, |state, cx| {
                        if state.auth_generation != 0 {
                            return None;
                        }
                        match result {
                            Ok(accounts) => {
                                state.auth_generation = state.auth_generation.wrapping_add(1);
                                state.accounts = accounts;
                            }
                            Err(error) => {
                                log::error!("Failed to load ChatGPT accounts: {error:#}");
                                state.last_auth_error =
                                    Some("Failed to load saved ChatGPT accounts.".into());
                            }
                        }
                        state
                            .is_authenticated()
                            .then(|| state.refresh_model_catalog(cx))
                    })?;
                    if let Some(refresh_models_task) = refresh_models_task
                        && let Err(error) = refresh_models_task.await
                    {
                        log::warn!("Failed to refresh ChatGPT models: {error:#}");
                    }
                    this.update(cx, |state, cx| {
                        state.load_task = None;
                        cx.notify();
                    })?;
                    Ok::<(), Arc<anyhow::Error>>(())
                }
            })
            .shared();

        Self {
            accounts: Accounts::default(),
            usage_requests: BTreeMap::new(),
            configured_policy: None,
            persistence_task: None,
            sign_in_state: SignInState::Idle,
            refresh_tasks: BTreeMap::new(),
            load_task: Some(load_task),
            credentials_provider,
            http_client,
            client_version: UNGATED_MODEL_CATALOG_CLIENT_VERSION.into(),
            available_models: ChatGptModel::all(),
            auth_generation: 0,
            model_catalog_generation: 0,
            last_auth_error: None,
            last_model_catalog_error: None,
        }
    }

    pub fn is_authenticated(&self) -> bool {
        self.accounts
            .accounts
            .values()
            .any(|account| account.credentials.is_some() && !account.needs_reauth)
    }

    pub fn email(&self) -> Option<&str> {
        self.accounts
            .accounts
            .get(&self.accounts.selected)
            .and_then(|account| account.credentials.as_ref())
            .and_then(|credentials| credentials.email.as_deref())
    }

    pub fn accounts(&self) -> Vec<language_model::LanguageModelAccount> {
        self.accounts
            .accounts
            .iter()
            .map(|(id, account)| language_model::LanguageModelAccount {
                id: id.clone(),
                label: account
                    .credentials
                    .as_ref()
                    .and_then(|credentials| credentials.email.clone())
                    .unwrap_or_else(|| id.clone()),
                selected: id == &self.accounts.selected,
                status: if account.needs_reauth {
                    "Sign in again".into()
                } else if account.billing_blocked {
                    "Billing requires attention".into()
                } else if account.quota_until_ms > now_ms() {
                    "Plan quota reached".into()
                } else if (account.status == "Rate limited"
                    && account.unavailable_until_ms <= now_ms())
                    || (account.status == "Plan quota reached"
                        && account.quota_until_ms <= now_ms())
                {
                    String::new()
                } else {
                    account.status.clone()
                },
            })
            .collect()
    }

    pub fn switch_policy(&self) -> AccountSwitchPolicy {
        self.configured_policy.unwrap_or_default()
    }

    pub fn configure_switch_policy(
        &mut self,
        policy: Option<AccountSwitchPolicy>,
        cx: &mut Context<Self>,
    ) {
        self.configured_policy = policy;
        cx.notify();
    }

    pub fn select_account(&mut self, id: String, cx: &mut Context<Self>) {
        if !self.accounts.accounts.contains_key(&id) {
            return;
        }
        self.accounts.selected = id;
        self.reset_model_catalog();
        self.persist(cx).detach_and_log_err(cx);
        self.refresh_model_catalog(cx).detach_and_log_err(cx);
        cx.notify();
    }

    pub fn reset_account_availability(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(account) = self.accounts.accounts.get_mut(id) {
            if account.needs_reauth {
                return;
            }
            account.quota_until_ms = 0;
            if account.status == "Plan quota reached" {
                account.status.clear();
            }
            self.persist(cx).detach_and_log_err(cx);
            cx.notify();
        }
    }

    pub fn remove_account(&mut self, id: &str, cx: &mut Context<Self>) {
        self.usage_requests.remove(id);
        self.accounts.accounts.remove(id);
        self.refresh_tasks.remove(id);
        self.auth_generation = self.auth_generation.wrapping_add(1);
        if self.accounts.selected == id {
            self.accounts.selected = self
                .accounts
                .accounts
                .keys()
                .next()
                .cloned()
                .unwrap_or_default();
            self.reset_model_catalog();
            if self.is_authenticated() {
                self.refresh_model_catalog(cx).detach_and_log_err(cx);
            }
        }
        self.persist(cx).detach_and_log_err(cx);
        cx.notify();
    }

    fn persist(&mut self, cx: &mut Context<Self>) -> Task<Result<(), Arc<anyhow::Error>>> {
        let previous = self.persistence_task.take();
        let accounts = self.accounts.clone();
        let provider = self.credentials_provider.clone();
        let task = cx
            .spawn(async move |this, cx| {
                if let Some(previous) = previous {
                    previous.await.log_err();
                }
                let result = save_accounts(provider.as_ref(), &accounts, cx).await;
                if result.is_err() {
                    this.update(cx, |state, cx| {
                        state.last_auth_error =
                            Some("Failed to save ChatGPT accounts. Please try again.".into());
                        cx.notify();
                    })
                    .log_err();
                }
                result.map_err(Arc::new)
            })
            .shared();
        self.persistence_task = Some(task.clone());
        cx.spawn(async move |_, _| task.await)
    }

    pub fn is_signing_in(&self) -> bool {
        !matches!(self.sign_in_state, SignInState::Idle)
    }

    pub fn is_sign_in_cancellable(&self) -> bool {
        matches!(self.sign_in_state, SignInState::Authorizing(_))
    }

    fn begin_persisting_credentials(&mut self, cx: &mut Context<Self>) {
        let sign_in_state = std::mem::replace(&mut self.sign_in_state, SignInState::Idle);
        self.sign_in_state = match sign_in_state {
            SignInState::Authorizing(task) => {
                cx.notify();
                SignInState::PersistingCredentials { _task: task }
            }
            sign_in_state => sign_in_state,
        };
    }

    pub fn last_auth_error(&self) -> Option<SharedString> {
        self.last_auth_error.clone()
    }

    pub fn model_catalog_error(&self) -> Option<SharedString> {
        self.last_model_catalog_error.clone()
    }

    pub fn available_models(&self) -> &[ChatGptModel] {
        &self.available_models
    }

    pub fn default_model(&self) -> Option<ChatGptModel> {
        self.available_models.first().cloned()
    }

    pub fn default_fast_model(&self) -> Option<ChatGptModel> {
        self.available_models
            .iter()
            .find(|model| model.id() == "gpt-5.6-luna")
            .cloned()
    }

    /// The in-flight task loading persisted credentials, or `None` once the
    /// initial load has finished.
    pub fn load_task(&self) -> Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>> {
        self.load_task.clone()
    }

    fn refresh_model_catalog(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Task<Result<(), Arc<anyhow::Error>>> {
        self.model_catalog_generation = self.model_catalog_generation.wrapping_add(1);
        let model_catalog_generation = self.model_catalog_generation;
        let http_client = self.http_client.clone();
        let client_version = self.client_version.clone();

        cx.spawn(async move |this, cx| {
            let result = async {
                let credentials = get_fresh_credentials(&this, &http_client, cx)
                    .await
                    .map_err(|error| anyhow!("{error}"))?;
                let request =
                    list_models(http_client.as_ref(), &credentials, client_version.as_ref());
                let timeout = cx
                    .background_executor()
                    .timer(MODEL_CATALOG_REQUEST_TIMEOUT);
                futures::select! {
                    result = request.fuse() => result,
                    () = timeout.fuse() => Err(anyhow!(
                        "ChatGPT models request timed out after {MODEL_CATALOG_REQUEST_TIMEOUT:?}"
                    )),
                }
            }
            .await;

            match result {
                Ok(models) => this
                    .update(cx, |state, cx| {
                        if state.model_catalog_generation == model_catalog_generation {
                            state.available_models = models;
                            state.last_model_catalog_error = None;
                            cx.notify();
                        }
                    })
                    .map_err(Arc::new),
                Err(error) => {
                    let error = Arc::new(error);
                    this.update(cx, |state, cx| {
                        if state.model_catalog_generation == model_catalog_generation {
                            state.last_model_catalog_error =
                                Some(format!("Failed to load models: {error:#}").into());
                            cx.notify();
                        }
                    })
                    .map_err(Arc::new)?;
                    Err(error)
                }
            }
        })
    }

    fn reset_model_catalog(&mut self) {
        self.model_catalog_generation = self.model_catalog_generation.wrapping_add(1);
        self.available_models = ChatGptModel::all();
        self.last_model_catalog_error = None;
    }

    /// Starts the browser-based OAuth sign-in flow. No-op while a sign-in is
    /// already in progress; observe the entity to react to the outcome.
    pub fn sign_in(&mut self, cx: &mut Context<Self>) {
        if self.is_signing_in() {
            return;
        }

        let http_client = self.http_client.clone();
        let load_task = self.load_task.clone();
        let task = cx.spawn(async move |this, cx| {
            if let Some(load_task) = load_task { load_task.await.map_err(|error| anyhow!("{error}"))?; }
            match do_oauth_flow(http_client, cx).await {
                Ok(creds) => {
                    this.update(cx, |state, cx| {
                        state.begin_persisting_credentials(cx);
                    })?;

                    let persist_result = this.update(cx, |state, cx| {
                        state.auth_generation = state.auth_generation.wrapping_add(1);
                        let id = creds.account_id.clone().or_else(|| creds.email.clone()).unwrap_or_else(|| "legacy".into());
                        state.refresh_tasks.remove(&id);
                        state.usage_requests.remove(&id);
                        state.accounts.insert(creds);
                        state.persist(cx)
                    })?.await;

                    match persist_result {
                        Ok(()) => {
                            let refresh_models_task = this.update(cx, |state, cx| {
                                state.auth_generation = state.auth_generation.wrapping_add(1);
                                state.last_auth_error = None;
                                state.refresh_model_catalog(cx)
                            })?;
                            if let Err(error) = refresh_models_task.await {
                                log::warn!("Failed to refresh ChatGPT models: {error:#}");
                            }
                            this.update(cx, |state, cx| {
                                state.sign_in_state = SignInState::Idle;
                                cx.notify();
                            })?;
                        }
                        Err(err) => {
                            log::error!(
                                "ChatGPT subscription sign-in failed to persist credentials: {err:?}"
                            );
                            this.update(cx, |state, cx| {
                                state.sign_in_state = SignInState::Idle;
                                state.last_auth_error =
                                    Some("Failed to save credentials. Please try again.".into());
                                cx.notify();
                            })
                            .log_err();
                        }
                    }
                }
                Err(err) => {
                    log::error!("ChatGPT subscription sign-in failed: {err:?}");
                    this.update(cx, |state, cx| {
                        state.sign_in_state = SignInState::Idle;
                        state.last_auth_error = Some("Sign-in failed. Please try again.".into());
                        cx.notify();
                    })
                    .log_err();
                }
            }
            anyhow::Ok(())
        });

        self.last_auth_error = None;
        self.sign_in_state = SignInState::Authorizing(task);
        cx.notify();
    }

    pub fn cancel_sign_in(&mut self, cx: &mut Context<Self>) {
        if matches!(self.sign_in_state, SignInState::Authorizing(_)) {
            self.sign_in_state = SignInState::Idle;
            cx.notify();
        }
    }

    /// Clears credentials and in-flight work immediately (so observers see the
    /// sign-out right away); the returned task deletes the persisted
    /// credentials.
    pub fn sign_out(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        self.auth_generation += 1;
        self.usage_requests.clear();
        self.accounts = Accounts::default();
        self.sign_in_state = SignInState::Idle;
        self.refresh_tasks.clear();
        self.last_auth_error = None;
        self.reset_model_catalog();
        cx.notify();
        let persist = self.persist(cx);
        cx.spawn(async move |_this, _cx| persist.await.map_err(|error| anyhow!("{error}")))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ChatGptModel {
    Gpt56Sol,
    Gpt56Terra,
    Gpt56Luna,
    Gpt55,
    Gpt54,
    Gpt54Mini,
    Discovered(Box<DiscoveredChatGptModel>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveredChatGptModel {
    id: String,
    display_name: String,
    max_token_count: u64,
    supports_images: bool,
    default_reasoning_effort: Option<ReasoningEffort>,
    supported_reasoning_efforts: Vec<ReasoningEffort>,
    supports_priority: bool,
}

impl ChatGptModel {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Gpt56Sol,
            Self::Gpt56Terra,
            Self::Gpt56Luna,
            Self::Gpt55,
            Self::Gpt54,
            Self::Gpt54Mini,
        ]
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Gpt56Sol => "gpt-5.6-sol",
            Self::Gpt56Terra => "gpt-5.6-terra",
            Self::Gpt56Luna => "gpt-5.6-luna",
            Self::Gpt55 => "gpt-5.5",
            Self::Gpt54 => "gpt-5.4",
            Self::Gpt54Mini => "gpt-5.4-mini",
            Self::Discovered(model) => &model.id,
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Gpt56Sol => "GPT-5.6 Sol",
            Self::Gpt56Terra => "GPT-5.6 Terra",
            Self::Gpt56Luna => "GPT-5.6 Luna",
            Self::Gpt55 => "GPT-5.5",
            Self::Gpt54 => "GPT-5.4",
            Self::Gpt54Mini => "GPT-5.4 Mini",
            Self::Discovered(model) => &model.display_name,
        }
    }

    fn max_token_count(&self) -> u64 {
        match self {
            Self::Gpt56Sol | Self::Gpt56Terra | Self::Gpt56Luna => 372_000,
            Self::Gpt55 | Self::Gpt54 | Self::Gpt54Mini => 272_000,
            Self::Discovered(model) => model.max_token_count,
        }
    }

    fn max_output_tokens(&self) -> Option<u64> {
        // Codex model metadata does not expose a max output token cap for these
        // models. Source: openai/codex models-manager/models.json.
        None
    }

    fn supports_images(&self) -> bool {
        match self {
            Self::Discovered(model) => model.supports_images,
            _ => true,
        }
    }

    fn default_reasoning_effort(&self) -> Option<ReasoningEffort> {
        match self {
            Self::Gpt56Sol => Some(ReasoningEffort::Low),
            Self::Gpt56Terra | Self::Gpt56Luna | Self::Gpt55 | Self::Gpt54 | Self::Gpt54Mini => {
                Some(ReasoningEffort::Medium)
            }
            Self::Discovered(model) => model.default_reasoning_effort,
        }
    }

    fn supported_reasoning_efforts(&self) -> &[ReasoningEffort] {
        match self {
            Self::Gpt56Sol | Self::Gpt56Terra | Self::Gpt56Luna => &[
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::XHigh,
                ReasoningEffort::Max,
            ],
            Self::Gpt55 | Self::Gpt54 | Self::Gpt54Mini => &[
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::XHigh,
            ],
            Self::Discovered(model) => &model.supported_reasoning_efforts,
        }
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn supports_prompt_cache_key(&self) -> bool {
        true
    }

    pub fn supports_priority(&self) -> bool {
        match self {
            Self::Gpt56Sol | Self::Gpt56Terra | Self::Gpt56Luna | Self::Gpt55 | Self::Gpt54 => true,
            Self::Gpt54Mini => false,
            Self::Discovered(model) => model.supports_priority,
        }
    }
}

/// Creates a [`LanguageModel`] for `model` that authenticates through
/// `state`'s credentials, refreshing them as needed.
pub fn create_language_model(
    model: ChatGptModel,
    state: &Entity<State>,
    cx: &App,
) -> Arc<dyn LanguageModel> {
    Arc::new(OpenAiSubscribedLanguageModel {
        id: LanguageModelId::from(model.id().to_string()),
        http_client: state.read(cx).http_client.clone(),
        model,
        state: state.clone(),
        request_limiter: RateLimiter::new(4),
        account_id: None,
    })
}

#[derive(Clone)]
struct OpenAiSubscribedLanguageModel {
    account_id: Option<String>,
    id: LanguageModelId,
    model: ChatGptModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl OpenAiSubscribedLanguageModel {
    fn codex_responses_request(
        &self,
        mut request: LanguageModelRequest,
    ) -> Result<open_ai::responses::Request> {
        if !self.model.supports_priority() {
            request.speed = None;
        }
        let mut responses_request = into_open_ai_response(
            request,
            self.model.id(),
            self.model.supports_parallel_tool_calls(),
            self.model.supports_prompt_cache_key(),
            None,
            self.model.default_reasoning_effort(),
            self.model
                .supported_reasoning_efforts()
                .contains(&ReasoningEffort::None),
            &PROVIDER_ID,
        )?;
        responses_request.store = Some(false);
        responses_request.instructions.get_or_insert_default();
        Ok(responses_request)
    }
}

fn codex_extra_headers(
    credentials: &CodexCredentials,
    routing_cache_key: Option<&str>,
) -> CustomHeaders {
    let mut header_pairs: Vec<(HeaderName, HeaderValue)> = vec![
        (
            HeaderName::from_static("originator"),
            HeaderValue::from_static("zed"),
        ),
        (
            HeaderName::from_static("openai-beta"),
            HeaderValue::from_static("responses=experimental"),
        ),
    ];
    if let Some(id) = &credentials.account_id
        && !id.is_empty()
        && let Ok(value) = HeaderValue::from_str(id)
    {
        header_pairs.push((HeaderName::from_static("chatgpt-account-id"), value));
    }
    if let Some(routing_cache_key) = routing_cache_key
        && let Ok(value) = HeaderValue::from_str(routing_cache_key)
    {
        header_pairs.push((HeaderName::from_static("session-id"), value.clone()));
        header_pairs.push((HeaderName::from_static("thread-id"), value));
    }
    CustomHeaders::new(header_pairs)
}

#[derive(Deserialize)]
struct ModelsResponse {
    models: Vec<CatalogModel>,
}

#[derive(Deserialize)]
struct CatalogModel {
    slug: String,
    display_name: String,
    default_reasoning_level: Option<String>,
    #[serde(default)]
    supported_reasoning_levels: Vec<ReasoningEffortPreset>,
    visibility: ModelVisibility,
    priority: i32,
    #[serde(default)]
    additional_speed_tiers: Vec<String>,
    #[serde(default)]
    service_tiers: Vec<ModelServiceTier>,
    context_window: Option<u64>,
    max_context_window: Option<u64>,
    input_modalities: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct ReasoningEffortPreset {
    effort: String,
}

#[derive(Deserialize)]
struct ModelServiceTier {
    id: String,
}

#[derive(Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ModelVisibility {
    List,
    Hide,
    None,
}

impl From<CatalogModel> for ChatGptModel {
    fn from(model: CatalogModel) -> Self {
        let default_reasoning_effort = model
            .default_reasoning_level
            .as_deref()
            .and_then(parse_reasoning_effort);
        let mut supported_reasoning_efforts = model
            .supported_reasoning_levels
            .iter()
            .filter_map(|preset| parse_reasoning_effort(&preset.effort))
            .collect::<Vec<_>>();
        if supported_reasoning_efforts.is_empty()
            && let Some(default_reasoning_effort) = default_reasoning_effort
        {
            supported_reasoning_efforts.push(default_reasoning_effort);
        }
        let supports_priority = model
            .additional_speed_tiers
            .iter()
            .any(|tier| tier == "fast")
            || model.service_tiers.iter().any(|tier| tier.id == "priority");
        let supports_images = model
            .input_modalities
            .as_ref()
            .is_none_or(|modalities| modalities.iter().any(|modality| modality == "image"));

        Self::Discovered(Box::new(DiscoveredChatGptModel {
            id: model.slug,
            display_name: model.display_name,
            max_token_count: model
                .context_window
                .or(model.max_context_window)
                .unwrap_or(272_000),
            supports_images,
            default_reasoning_effort,
            supported_reasoning_efforts,
            supports_priority,
        }))
    }
}

fn parse_reasoning_effort(effort: &str) -> Option<ReasoningEffort> {
    match effort {
        "none" => Some(ReasoningEffort::None),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

async fn list_models(
    http_client: &dyn HttpClient,
    credentials: &CodexCredentials,
    client_version: &str,
) -> Result<Vec<ChatGptModel>> {
    let query = form_urlencoded::Serializer::new(String::new())
        .append_pair("client_version", client_version)
        .finish();
    let uri = format!("{CODEX_BASE_URL}/models?{query}");
    let extra_headers = codex_extra_headers(credentials, None);
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(uri)
        .header("Accept", "application/json")
        .header(
            "Authorization",
            format!("Bearer {}", credentials.access_token),
        )
        .extra_headers(&extra_headers)
        .body(AsyncBody::default())
        .context("failed to build ChatGPT models request")?;
    let mut response = http_client
        .send(request)
        .await
        .context("failed to request ChatGPT models")?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body)
        .await
        .context("failed to read ChatGPT models response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "ChatGPT models request failed (HTTP {status}): {body}"
        ));
    }

    let mut models = serde_json::from_str::<ModelsResponse>(&body)
        .context("failed to parse ChatGPT models response")?
        .models;
    models.retain(|model| model.visibility == ModelVisibility::List);
    models.sort_by_key(|model| model.priority);
    if models.is_empty() {
        return Err(anyhow!(
            "ChatGPT models response did not contain any picker-visible models"
        ));
    }
    Ok(models.into_iter().map(ChatGptModel::from).collect())
}

impl LanguageModel for OpenAiSubscribedLanguageModel {
    fn account_usage(
        &self,
        force: bool,
        cx: &mut App,
    ) -> Option<Task<Result<language_model::LanguageModelAccountUsage>>> {
        let id = self
            .account_id
            .clone()
            .unwrap_or_else(|| self.state.read(cx).accounts.selected.clone());
        Some(
            self.state
                .update(cx, |state, cx| usage::load(state, id, force, cx)),
        )
    }

    fn account_id(&self) -> Option<&str> {
        self.account_id.as_deref()
    }
    fn accounts(&self, cx: &App) -> Vec<language_model::LanguageModelAccount> {
        self.state
            .read(cx)
            .accounts()
            .into_iter()
            .map(|mut account| {
                if let Some(id) = &self.account_id {
                    account.selected = &account.id == id;
                }
                account
            })
            .collect()
    }

    fn with_account(&self, id: String) -> Option<Arc<dyn LanguageModel>> {
        Some(Arc::new(Self {
            account_id: Some(id),
            ..self.clone()
        }))
    }

    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        true
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn supports_thinking(&self) -> bool {
        true
    }

    fn supports_fast_mode(&self) -> bool {
        self.model.supports_priority()
    }

    fn supports_server_side_compaction(&self) -> bool {
        true
    }

    fn supports_explicit_compaction(&self) -> bool {
        true
    }

    fn compact(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<CompactionResult, LanguageModelCompletionError>> {
        let mut responses_request = match self.codex_responses_request(request) {
            Ok(responses_request) => responses_request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        responses_request.context_management = None;
        responses_request
            .input
            .push(ResponseInputItem::CompactionTrigger);

        let state = self.state.downgrade();
        let http_client = self.http_client.clone();
        let request_limiter = self.request_limiter.clone();
        let account_id = self.account_id.clone();

        cx.spawn(async move |cx| {
            let response_stream = request_limiter.stream(
                stream_with_accounts(&state, &http_client, &responses_request, account_id.as_deref(), cx)
            ).await?;
            let mapper = OpenAiResponseEventMapper::new(PROVIDER_ID);
            let mut event_stream = language_model::stream_in_background(
                mapper.map_stream(response_stream.boxed()).boxed(),
                cx.background_executor().clone(),
            );
            let mut compacted_context = None;
            let mut usage = language_model::TokenUsage::default();

            while let Some(event) = event_stream.next().await {
                match event? {
                    LanguageModelCompletionEvent::Compaction(
                        language_model::CompactionUpdate::Finished(context),
                    ) => {
                        if compacted_context.replace(context).is_some() {
                            return Err(LanguageModelCompletionError::Other(anyhow!(
                                "ChatGPT subscription compaction returned multiple replacement contexts"
                            )));
                        }
                    }
                    LanguageModelCompletionEvent::UsageUpdate(updated_usage) => {
                        usage = updated_usage;
                    }
                    _ => {}
                }
            }

            let context = compacted_context.ok_or_else(|| {
                LanguageModelCompletionError::Other(anyhow!(
                    "ChatGPT subscription compaction returned no replacement context"
                ))
            })?;
            Ok(CompactionResult { context, usage })
        })
        .boxed()
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        let default_effort = self.model.default_reasoning_effort();
        self.model
            .supported_reasoning_efforts()
            .iter()
            .copied()
            .filter_map(|effort| {
                let (name, value) = match effort {
                    ReasoningEffort::None => return None,
                    ReasoningEffort::Minimal => ("Minimal", "minimal"),
                    ReasoningEffort::Low => ("Low", "low"),
                    ReasoningEffort::Medium => ("Medium", "medium"),
                    ReasoningEffort::High => ("High", "high"),
                    ReasoningEffort::XHigh => ("Extra High", "xhigh"),
                    ReasoningEffort::Max => ("Max", "max"),
                };

                Some(LanguageModelEffortLevel {
                    name: name.into(),
                    value: value.into(),
                    is_default: Some(effort) == default_effort,
                })
            })
            .collect()
    }

    fn telemetry_id(&self) -> String {
        format!("openai-subscribed/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let responses_request = match self.codex_responses_request(request) {
            Ok(responses_request) => responses_request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };

        let state = self.state.downgrade();
        let http_client = self.http_client.clone();
        let request_limiter = self.request_limiter.clone();
        let account_id = self.account_id.clone();
        let executor = cx.background_executor().clone();

        let future = cx.spawn(async move |cx| {
            request_limiter
                .stream(stream_with_accounts(
                    &state,
                    &http_client,
                    &responses_request,
                    account_id.as_deref(),
                    cx,
                ))
                .await
        });

        async move {
            let mapper = OpenAiResponseEventMapper::new(PROVIDER_ID);
            Ok(language_model::stream_in_background(
                mapper.map_stream(future.await?.boxed()).boxed(),
                executor,
            ))
        }
        .boxed()
    }
}

#[derive(Debug, PartialEq)]
enum AccountFailure {
    Unauthorized,
    NeedsReauth,
    Scope(Option<String>),
    RateLimit(u64),
    Quota(u64),
    Billing,
    Transient,
    Other,
}

fn classify_failure(error: &open_ai::RequestError) -> AccountFailure {
    let open_ai::RequestError::HttpResponseError {
        status_code,
        body,
        headers,
        ..
    } = error
    else {
        return if matches!(
            error,
            open_ai::RequestError::HttpSend { .. } | open_ai::RequestError::ReadResponse { .. }
        ) || matches!(error, open_ai::RequestError::Other(error) if error.downcast_ref::<std::io::Error>().is_some())
        {
            AccountFailure::Transient
        } else {
            AccountFailure::Other
        };
    };
    let body_lower = body.to_ascii_lowercase();
    let payload = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    let detail = payload.get("error").unwrap_or(&payload);
    let reset = detail
        .get("resets_at")
        .or_else(|| detail.get("reset_at"))
        .and_then(|value| value.as_u64())
        .map(|seconds| seconds.saturating_mul(1000))
        .filter(|reset| *reset > now_ms());
    if matches!(status_code.as_u16(), 403 | 429)
        && [
            "usage_limit_reached",
            "usage limit",
            "plan limit",
            "limit reached on your plan",
            "quota",
            "usage cap",
        ]
        .iter()
        .any(|signal| body_lower.contains(signal))
    {
        return AccountFailure::Quota(reset.unwrap_or(u64::MAX));
    }
    if status_code.as_u16() == 402
        || (matches!(status_code.as_u16(), 403 | 429)
            && ["billing", "no credit", "no_credit", "plan expired"]
                .iter()
                .any(|signal| body_lower.contains(signal)))
    {
        return AccountFailure::Billing;
    }
    if ["consent_required", "interaction_required", "login_required"]
        .iter()
        .any(|signal| body_lower.contains(signal))
    {
        return AccountFailure::NeedsReauth;
    }
    match status_code.as_u16() {
        401 => AccountFailure::Unauthorized,
        403 => AccountFailure::Scope(
            detail
                .get("required_scope")
                .and_then(|value| value.as_str())
                .or_else(|| {
                    headers
                        .get("www-authenticate")?
                        .to_str()
                        .ok()?
                        .split("scope=\"")
                        .nth(1)?
                        .split('"')
                        .next()
                })
                .filter(|scope| !scope.trim().is_empty())
                .map(str::to_owned),
        ),
        429 => {
            let delay = headers
                .get("retry-after")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(30);
            AccountFailure::RateLimit(now_ms().saturating_add(delay.max(1).saturating_mul(1000)))
        }
        500..=599 => AccountFailure::Transient,
        _ => AccountFailure::Other,
    }
}

fn stream_event_error(event: &open_ai::responses::StreamEvent) -> Option<open_ai::RequestError> {
    use open_ai::responses::StreamEvent;
    let error = match event {
        StreamEvent::Error { error } => error.clone(),
        StreamEvent::Failed { response } => response.error.clone()?,
        StreamEvent::GenericError { error } => error.clone().into_response_error(),
        _ => return None,
    };
    let signal = format!(
        "{} {} {}",
        error.code.as_deref().unwrap_or_default(),
        error.error_type.as_deref().unwrap_or_default(),
        error.message
    )
    .to_ascii_lowercase();
    let status = if ["invalid_token", "expired_token", "authentication"]
        .iter()
        .any(|code| signal.contains(code))
    {
        401
    } else if ["billing", "no credit"]
        .iter()
        .any(|code| signal.contains(code))
    {
        402
    } else if [
        "scope",
        "permission",
        "consent",
        "login_required",
        "interaction_required",
    ]
    .iter()
    .any(|code| signal.contains(code))
    {
        403
    } else if ["limit", "quota", "slow_down"]
        .iter()
        .any(|code| signal.contains(code))
    {
        429
    } else if ["server_error", "timeout", "overloaded"]
        .iter()
        .any(|code| signal.contains(code))
    {
        503
    } else {
        400
    };
    Some(open_ai::RequestError::HttpResponseError {
        provider: PROVIDER_NAME.0.to_string(),
        status_code: http_client::StatusCode::from_u16(status).ok()?,
        body: serde_json::json!({ "error": error }).to_string(),
        headers: Box::default(),
    })
}

async fn preflight_stream(
    mut stream: futures::stream::BoxStream<'static, Result<open_ai::responses::StreamEvent>>,
    cx: &AsyncApp,
) -> Result<
    futures::stream::BoxStream<'static, Result<open_ai::responses::StreamEvent>>,
    open_ai::RequestError,
> {
    use open_ai::responses::StreamEvent;
    let mut buffered = Vec::new();
    loop {
        let timeout = cx.background_executor().timer(Duration::from_secs(60));
        let event = futures::select! {
            event = stream.next().fuse() => event,
            () = timeout.fuse() => return Err(open_ai::RequestError::HttpSend { provider: PROVIDER_NAME.0.to_string(), host: "chatgpt.com".into(), error: anyhow!("Response stream timed out") }),
        };
        let Some(event) = event else {
            return Ok(futures::stream::iter(buffered).boxed());
        };
        let event = event.map_err(open_ai::RequestError::Other)?;
        if let Some(error) = stream_event_error(&event) {
            return Err(error);
        }
        let metadata = matches!(
            event,
            StreamEvent::Created { .. } | StreamEvent::InProgress { .. } | StreamEvent::Unknown
        );
        buffered.push(Ok(event));
        // Once output has started, replay could duplicate text or tool calls.
        if !metadata || buffered.len() >= 32 {
            return Ok(futures::stream::iter(buffered).chain(stream).boxed());
        }
    }
}

async fn record_account_failure(
    state: &WeakEntity<State>,
    id: &str,
    model: &str,
    token: &str,
    failure: AccountFailure,
    cx: &mut AsyncApp,
) -> Result<(), LanguageModelCompletionError> {
    let persist = state.update(cx, |state, cx| {
        let Some(account) = state.accounts.accounts.get_mut(id) else {
            return None;
        };
        if matches!(
            failure,
            AccountFailure::Unauthorized | AccountFailure::NeedsReauth
        ) && account
            .credentials
            .as_ref()
            .is_some_and(|credentials| credentials.access_token != token)
        {
            return None;
        }
        match failure {
            AccountFailure::Unauthorized | AccountFailure::NeedsReauth => {
                account.needs_reauth = true;
                account.status = "Sign in again".into();
            }
            AccountFailure::Scope(_) => {
                if !account
                    .blocked_models
                    .iter()
                    .any(|blocked| blocked == model)
                {
                    account.blocked_models.push(model.to_owned());
                }
                account.status = format!("Missing permission for {model}");
            }
            AccountFailure::RateLimit(until) => {
                account.unavailable_until_ms = account.unavailable_until_ms.max(until);
                account.status = "Rate limited".into();
            }
            AccountFailure::Quota(until) => {
                account.quota_until_ms = account.quota_until_ms.max(until);
                account.status = "Plan quota reached".into();
            }
            AccountFailure::Billing => {
                account.billing_blocked = true;
                account.status = "Billing requires attention".into();
            }
            AccountFailure::Transient | AccountFailure::Other => return None,
        }
        cx.notify();
        Some(state.persist(cx))
    })?;
    if let Some(persist) = persist {
        persist.await.map_err(|error| anyhow!("{error}"))?;
    }
    Ok(())
}

fn monitor_account_stream(
    state: WeakEntity<State>,
    http_client: Arc<dyn HttpClient>,
    id: String,
    model: String,
    token: String,
    mut stream: futures::stream::BoxStream<'static, Result<open_ai::responses::StreamEvent>>,
    cx: &AsyncApp,
) -> futures::stream::BoxStream<'static, Result<open_ai::responses::StreamEvent>> {
    let (mut sender, receiver) = futures::channel::mpsc::channel(1);
    let task = cx.spawn(async move |cx| {
        loop {
            let timeout = cx.background_executor().timer(Duration::from_secs(60));
            let event = futures::select! {
                event = stream.next().fuse() => event,
                () = timeout.fuse() => {
                    if sender.send(Err(anyhow!("ChatGPT response stream timed out"))).await.is_err() { return; }
                    break;
                }
            };
            let Some(event) = event else { break; };
            let failure = event
                .as_ref()
                .ok()
                .and_then(stream_event_error)
                .map(|error| classify_failure(&error));
            let failed = event.is_err() || failure.is_some();
            if let Some(failure) = failure {
                let result = if failure == AccountFailure::Unauthorized {
                    account_credentials(&state, &http_client, &id, Some(&token), cx)
                        .await
                        .map(|_| ())
                } else {
                    record_account_failure(&state, &id, &model, &token, failure, cx).await
                };
                if let Err(error) = result {
                    if sender.send(Err(anyhow!("{error}"))).await.is_err() {
                        break;
                    }
                }
            }
            if sender.send(event).await.is_err() || failed {
                break;
            }
        }
    });
    futures::stream::unfold((receiver, task), |(mut receiver, task)| async move {
        receiver.next().await.map(|event| (event, (receiver, task)))
    })
    .boxed()
}

async fn stream_with_accounts(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    request: &open_ai::responses::Request,
    preferred: Option<&str>,
    cx: &mut AsyncApp,
) -> Result<
    futures::stream::BoxStream<'static, Result<open_ai::responses::StreamEvent>>,
    LanguageModelCompletionError,
> {
    let (mut candidates, policy) = state.read_with(cx, |state, _| {
        let preferred = preferred.unwrap_or(&state.accounts.selected);
        let mut candidates = vec![preferred.to_owned()];
        if state.switch_policy() == AccountSwitchPolicy::OnError {
            candidates.extend(
                state
                    .accounts
                    .accounts
                    .keys()
                    .filter(|id| id.as_str() != preferred)
                    .cloned(),
            );
        }
        (candidates, state.switch_policy())
    })?;
    let mut last_error = None;
    let mut required_scope: Option<String> = None;
    for id in candidates.drain(..) {
        let eligible = state.read_with(cx, |state, _| {
            state.accounts.accounts.get(&id).is_some_and(|account| {
                account.available(&request.model)
                    && required_scope.as_ref().is_none_or(|scope| {
                        account.credentials.as_ref().is_some_and(|credentials| {
                            scope.split_whitespace().all(|required| {
                                credentials.scopes.iter().any(|granted| granted == required)
                            })
                        })
                    })
            })
        })?;
        if !eligible {
            continue;
        }
        let mut credentials = match account_credentials(state, http_client, &id, None, cx).await {
            Ok(credentials) => credentials,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        let mut refreshed = false;
        let mut retried = false;
        loop {
            let headers = codex_extra_headers(&credentials, request.prompt_cache_key.as_deref());
            let provider_name = PROVIDER_NAME;
            let response = stream_response_ref(
                http_client.as_ref(),
                provider_name.0.as_str(),
                CODEX_BASE_URL,
                &credentials.access_token,
                request,
                &headers,
            );
            let timeout = cx.background_executor().timer(Duration::from_secs(60));
            let result = futures::select! {
                response = response.fuse() => response,
                () = timeout.fuse() => Err(open_ai::RequestError::HttpSend { provider: PROVIDER_NAME.0.to_string(), host: "chatgpt.com".into(), error: anyhow!("Request timed out") }),
            };
            let result = match result {
                Ok(stream) => preflight_stream(stream, cx).await,
                Err(error) => Err(error),
            };
            let error = match result {
                Ok(stream) => {
                    return Ok(monitor_account_stream(
                        state.clone(),
                        http_client.clone(),
                        id,
                        request.model.clone(),
                        credentials.access_token,
                        stream,
                        cx,
                    ));
                }
                Err(error) => error,
            };
            let failure = classify_failure(&error);
            if failure == AccountFailure::Unauthorized && !refreshed {
                refreshed = true;
                match account_credentials(
                    state,
                    http_client,
                    &id,
                    Some(&credentials.access_token),
                    cx,
                )
                .await
                {
                    Ok(fresh) => {
                        credentials = fresh;
                        continue;
                    }
                    Err(error) => {
                        last_error = Some(error);
                        break;
                    }
                }
            }
            if failure == AccountFailure::Transient && !retried {
                retried = true;
                cx.background_executor().timer(Duration::from_secs(1)).await;
                continue;
            }
            let can_failover = match &failure {
                AccountFailure::Other => false,
                AccountFailure::Scope(scope) => {
                    required_scope = scope.clone();
                    scope.is_some()
                }
                _ => true,
            };
            record_account_failure(
                state,
                &id,
                &request.model,
                &credentials.access_token,
                failure,
                cx,
            )
            .await?;
            if !can_failover || policy == AccountSwitchPolicy::Manual {
                return Err(error.into());
            }
            last_error = Some(error.into());
            break;
        }
    }
    Err(last_error.unwrap_or_else(|| {
        anyhow!(
            "No eligible ChatGPT account. Check account status in Settings > AI > LLM Providers."
        )
        .into()
    }))
}

async fn get_fresh_credentials(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    cx: &mut AsyncApp,
) -> Result<CodexCredentials, LanguageModelCompletionError> {
    let id = state.read_with(cx, |state, _| state.accounts.selected.clone())?;
    account_credentials(state, http_client, &id, None, cx).await
}

async fn refresh_with_timeout(
    http_client: &Arc<dyn HttpClient>,
    refresh_token_value: &str,
    cx: &AsyncApp,
) -> Result<CodexCredentials, RefreshError> {
    let timeout = cx.background_executor().timer(Duration::from_secs(60));
    futures::select! {
        result = refresh_token(http_client, refresh_token_value).fuse() => result,
        () = timeout.fuse() => Err(RefreshError::Transient(anyhow!("ChatGPT token refresh timed out"))),
    }
}

async fn account_credentials(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    id: &str,
    rejected_token: Option<&str>,
    cx: &mut AsyncApp,
) -> Result<CodexCredentials, LanguageModelCompletionError> {
    let (credentials, existing_task) = state.read_with(cx, |state, _| {
        (
            state
                .accounts
                .accounts
                .get(id)
                .filter(|account| !account.needs_reauth)
                .and_then(|account| account.credentials.clone()),
            state.refresh_tasks.get(id).cloned(),
        )
    })?;
    let credentials = credentials.ok_or(LanguageModelCompletionError::NoApiKey {
        provider: PROVIDER_NAME,
    })?;
    if let Some(task) = existing_task {
        return task.await.map_err(|error| anyhow!("{error}").into());
    }
    if !credentials.is_expired() && rejected_token != Some(credentials.access_token.as_str()) {
        return Ok(credentials);
    }
    let http_client = http_client.clone();
    let id = id.to_owned();
    let account_id = id.clone();
    let state_clone = state.clone();
    let task = cx
        .spawn(async move |cx| {
            let result = match refresh_with_timeout(&http_client, &credentials.refresh_token, cx)
                .await
            {
                Err(RefreshError::Transient(error)) => {
                    log::warn!("Retrying ChatGPT token refresh after a transient failure: {error}");
                    cx.background_executor().timer(Duration::from_secs(1)).await;
                    refresh_with_timeout(&http_client, &credentials.refresh_token, cx).await
                }
                result => result,
            };
            let (result, persist) = state_clone
                .update(cx, |state, cx| {
                    let Some(account) = state.accounts.accounts.get_mut(&account_id) else {
                        return (Err(Arc::new(anyhow!("Account was removed"))), None);
                    };
                    if account.credentials.as_ref().is_none_or(|current| {
                        current.refresh_token != credentials.refresh_token
                            || current.access_token != credentials.access_token
                    }) {
                        return (
                            Err(Arc::new(anyhow!("Account changed during token refresh"))),
                            None,
                        );
                    }
                    state.refresh_tasks.remove(&account_id);
                    let result = match result {
                        Ok(mut refreshed) => {
                            if refreshed
                                .account_id
                                .as_ref()
                                .zip(credentials.account_id.as_ref())
                                .is_some_and(|(new, previous)| new != previous)
                            {
                                account.needs_reauth = true;
                                account.status =
                                    "Account changed during refresh. Sign in again".into();
                                return (
                                    Err(Arc::new(anyhow!(
                                        "Token refresh returned a different ChatGPT account"
                                    ))),
                                    Some(state.persist(cx)),
                                );
                            }
                            refreshed.account_id = refreshed.account_id.or(credentials.account_id);
                            refreshed.email = refreshed.email.or(credentials.email);
                            if refreshed.scopes.is_empty() {
                                refreshed.scopes = credentials.scopes;
                            }
                            account.credentials = Some(refreshed.clone());
                            Ok(refreshed)
                        }
                        Err(RefreshError::Fatal(error)) => {
                            account.needs_reauth = true;
                            account.status = "Sign in again".into();
                            state.last_auth_error =
                                Some("Your session has expired. Please sign in again.".into());
                            Err(Arc::new(error))
                        }
                        Err(RefreshError::Transient(error)) => return (Err(Arc::new(error)), None),
                    };
                    cx.notify();
                    (result, Some(state.persist(cx)))
                })
                .map_err(Arc::new)?;
            if let Some(persist) = persist {
                persist.await?;
            }
            result
        })
        .shared();
    state.update(cx, |state, _| {
        state.refresh_tasks.insert(id, task.clone());
    })?;
    task.await.map_err(|error| anyhow!("{error}").into())
}

#[derive(Deserialize)]
struct TokenResponse {
    #[serde(default)]
    scope: Option<String>,
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    id_token: Option<String>,
    expires_in: u64,
    #[serde(default)]
    email: Option<String>,
}

// The OAuth client registered for `CLIENT_ID` (the Codex CLI's client) only allows
// `http://localhost:1455/auth/callback` and `http://localhost:1457/auth/callback`
// as redirect URIs; using anything else (different host, port, or path) causes
// auth.openai.com to reject the authorize request with a generic `unknown_error`
// before redirecting back. Keep these in sync with the Codex CLI's redirect URI
// allow-list (see codex-rs/login/src/server.rs in openai/codex).
const CODEX_CALLBACK_HOST: &str = "localhost";
const CODEX_CALLBACK_PORT: u16 = 1455;
const CODEX_CALLBACK_FALLBACK_PORT: u16 = 1457;
const CODEX_CALLBACK_PATH: &str = "/auth/callback";

async fn do_oauth_flow(
    http_client: Arc<dyn HttpClient>,
    cx: &AsyncApp,
) -> Result<CodexCredentials> {
    // Start the callback server FIRST so the redirect URI is ready
    let (redirect_uri, callback_rx) =
        oauth_callback_server::start_oauth_callback_server_with_config(
            oauth_callback_server::OAuthCallbackServerConfig {
                host: CODEX_CALLBACK_HOST,
                preferred_port: CODEX_CALLBACK_PORT,
                fallback_port: Some(CODEX_CALLBACK_FALLBACK_PORT),
                path: CODEX_CALLBACK_PATH,
            },
        )
        .context("Failed to start OAuth callback server")?;

    // PKCE verifier: 32 random bytes → base64url (no padding)
    let mut verifier_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    // PKCE challenge: SHA-256(verifier) → base64url
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize().as_slice());

    // CSRF state: 16 random bytes → hex string
    let mut state_bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut state_bytes);
    let oauth_state: String = state_bytes.iter().map(|b| format!("{b:02x}")).collect();

    let mut auth_url = url::Url::parse(OPENAI_AUTHORIZE_URL).expect("valid base URL");
    auth_url
        .query_pairs_mut()
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", &redirect_uri)
        // Deliberately excludes `api.connectors.read api.connectors.invoke`
        // (which Codex CLI requests): extra scopes inflate the
        // access-token JWT, and the serialized credentials must fit within
        // Windows Credential Manager's 2560-byte blob limit
        // (CRED_MAX_CREDENTIAL_BLOB_SIZE). See #58541.
        .append_pair("scope", "openid profile email offline_access")
        .append_pair("prompt", "select_account")
        .append_pair("response_type", "code")
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("state", &oauth_state)
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", "zed");

    // Open browser AFTER the listener is ready
    cx.update(|cx| cx.open_url(auth_url.as_str()));

    // Await the callback
    let callback = callback_rx
        .await
        .map_err(|_| anyhow!("OAuth callback was cancelled"))?
        .context("OAuth callback failed")?;

    // Validate CSRF state
    if callback.state != oauth_state {
        return Err(anyhow!("OAuth state mismatch"));
    }

    let tokens = exchange_code(&http_client, &callback.code, &verifier, &redirect_uri)
        .await
        .context("Token exchange failed")?;

    let jwt = tokens
        .id_token
        .as_deref()
        .unwrap_or(tokens.access_token.as_str());
    let claims = extract_jwt_claims(jwt);

    Ok(CodexCredentials {
        access_token: tokens.access_token,
        scopes: tokens
            .scope
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
        refresh_token: tokens.refresh_token,
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        account_id: claims.account_id,
        email: claims.email.or(tokens.email),
    })
}

async fn exchange_code(
    client: &Arc<dyn HttpClient>,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<TokenResponse> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "authorization_code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("code", code)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("code_verifier", verifier)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(OPENAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?;

    let mut response = client.send(request).await?;
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Token exchange failed (HTTP {}): {body}",
            response.status()
        ));
    }

    serde_json::from_str::<TokenResponse>(&body).context("Failed to parse token response")
}

async fn refresh_token(
    client: &Arc<dyn HttpClient>,
    refresh_token: &str,
) -> Result<CodexCredentials, RefreshError> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("refresh_token", refresh_token)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(OPENAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))
        .map_err(|e| RefreshError::Transient(e.into()))?;

    let mut response = client
        .send(request)
        .await
        .map_err(|e| RefreshError::Transient(e))?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body)
        .await
        .map_err(|e| RefreshError::Transient(e.into()))?;

    if !status.is_success() {
        let err = anyhow!("Token refresh failed (HTTP {}): {body}", status);
        // 400/401/403 indicate a revoked or invalid refresh token.
        // 5xx and other errors are treated as transient.
        if status == http_client::StatusCode::BAD_REQUEST
            || status == http_client::StatusCode::UNAUTHORIZED
            || status == http_client::StatusCode::FORBIDDEN
        {
            return Err(RefreshError::Fatal(err));
        }
        return Err(RefreshError::Transient(err));
    }

    let tokens: TokenResponse =
        serde_json::from_str(&body).map_err(|e| RefreshError::Transient(e.into()))?;
    let jwt = tokens
        .id_token
        .as_deref()
        .unwrap_or(tokens.access_token.as_str());
    let claims = extract_jwt_claims(jwt);

    Ok(CodexCredentials {
        access_token: tokens.access_token,
        scopes: tokens
            .scope
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
        refresh_token: tokens.refresh_token,
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        account_id: claims.account_id,
        email: claims.email.or(tokens.email),
    })
}

struct JwtClaims {
    account_id: Option<String>,
    email: Option<String>,
}

/// Extract claims from a JWT payload (base64url middle segment).
/// Extracts `chatgpt_account_id` from three possible locations (matching Roo Code's
/// implementation) and the `email` claim.
fn extract_jwt_claims(jwt: &str) -> JwtClaims {
    let Some(payload_b64) = jwt.split('.').nth(1) else {
        return JwtClaims {
            account_id: None,
            email: None,
        };
    };
    let Ok(payload) = URL_SAFE_NO_PAD.decode(payload_b64) else {
        return JwtClaims {
            account_id: None,
            email: None,
        };
    };
    let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload) else {
        return JwtClaims {
            account_id: None,
            email: None,
        };
    };

    let account_id = claims
        .get("chatgpt_account_id")
        .and_then(|v| v.as_str())
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(|v| v.get("chatgpt_account_id"))
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            claims
                .get("organizations")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|org| org.get("id"))
                .and_then(|v| v.as_str())
        })
        .map(|s| s.to_owned());

    let email = claims
        .get("email")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());

    JwtClaims { account_id, email }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|err| {
            log::error!("System clock is before UNIX epoch: {err}");
            0
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AppContext as _, TestAppContext};
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_account_failure_classification() {
        for (status, body, expected) in [
            (401, "invalid_token", AccountFailure::Unauthorized),
            (401, "interaction_required", AccountFailure::NeedsReauth),
            (403, "insufficient_scope", AccountFailure::Scope(None)),
            (
                429,
                "limit reached on your plan",
                AccountFailure::Quota(u64::MAX),
            ),
            (403, "usage_limit_reached", AccountFailure::Quota(u64::MAX)),
            (402, "payment required", AccountFailure::Billing),
            (500, "server error", AccountFailure::Transient),
            (400, "bad request", AccountFailure::Other),
        ] {
            let error = open_ai::RequestError::HttpResponseError {
                provider: "test".into(),
                status_code: http_client::StatusCode::from_u16(status).expect("valid status"),
                body: body.into(),
                headers: Box::default(),
            };
            assert_eq!(classify_failure(&error), expected, "{status}: {body}");
        }
    }

    #[test]
    fn test_legacy_credentials_and_quota_round_trip() {
        let legacy = serde_json::to_vec(&make_fresh_credentials()).expect("serialize");
        let mut accounts = Accounts::decode(&legacy).expect("legacy credentials migrate");
        assert_eq!(accounts.accounts.len(), 1);
        let account = accounts
            .accounts
            .get_mut(&accounts.selected)
            .expect("account");
        account.quota_until_ms = u64::MAX;
        let encoded = serde_json::to_vec(&accounts).expect("serialize accounts");
        let restored = Accounts::decode(&encoded).expect("restore accounts");
        assert!(
            !restored
                .accounts
                .get(&restored.selected)
                .expect("account")
                .available("gpt-5.4")
        );
    }

    #[test]
    fn test_sse_quota_reset_time_is_preserved() {
        let reset_seconds = now_ms() / 1000 + 3600;
        let event = serde_json::from_value::<open_ai::responses::StreamEvent>(serde_json::json!({
            "type": "error", "error": { "code": "usage_limit_reached", "message": "Plan quota", "resets_at": reset_seconds }
        })).expect("event");
        let error = stream_event_error(&event).expect("error");
        assert_eq!(
            classify_failure(&error),
            AccountFailure::Quota(reset_seconds * 1000)
        );
    }

    #[gpui::test(iterations = 20)]
    async fn test_accounts_refresh_independently(cx: &mut TestAppContext) {
        let calls = Arc::new(AtomicUsize::new(0));
        let http: Arc<dyn HttpClient> = FakeHttpClient::create({
            let calls = calls.clone();
            move |mut request| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    let mut body = String::new();
                    smol::io::AsyncReadExt::read_to_string(request.body_mut(), &mut body).await?;
                    let token = if body.contains("first_refresh") {
                        "first_access"
                    } else {
                        "second_access"
                    };
                    let mut response: serde_json::Value =
                        serde_json::from_str(&fake_token_response())?;
                    response["access_token"] = token.into();
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(AsyncBody::from(response.to_string()))?)
                }
            }
        });
        let provider = Arc::new(FakeCredentialsProvider::new());
        let mut credentials = make_expired_credentials();
        credentials.account_id = Some("first".into());
        credentials.refresh_token = "first_refresh".into();
        let state = make_state_with_credentials_provider(
            http.clone(),
            Some(credentials),
            provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            let mut credentials = make_expired_credentials();
            credentials.account_id = Some("second".into());
            credentials.refresh_token = "second_refresh".into();
            state.accounts.insert(credentials);
        });
        let first = cx.spawn({
            let state = state.downgrade();
            let http = http.clone();
            async move |mut cx| account_credentials(&state, &http, "first", None, &mut cx).await
        });
        let second = cx.spawn({
            let state = state.downgrade();
            async move |mut cx| account_credentials(&state, &http, "second", None, &mut cx).await
        });
        assert_eq!(
            first.await.expect("first refresh").access_token,
            "first_access"
        );
        assert_eq!(
            second.await.expect("second refresh").access_token,
            "second_access"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let restored = load_accounts(provider.as_ref(), &cx.to_async())
            .await
            .expect("load");
        for (id, expected) in [("first", "first_access"), ("second", "second_access")] {
            assert_eq!(
                restored
                    .accounts
                    .get(id)
                    .expect("account")
                    .credentials
                    .as_ref()
                    .expect("credentials")
                    .access_token,
                expected
            );
        }
    }

    #[gpui::test]
    async fn test_thread_account_binding_does_not_change_other_models(cx: &mut TestAppContext) {
        let http = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(AsyncBody::from("data: [DONE]\n\n"))?)
        });
        let mut credentials = make_fresh_credentials();
        credentials.account_id = Some("first".into());
        let state = make_state(http, Some(credentials), cx);
        state.update(cx, |state, _| {
            let mut credentials = make_fresh_credentials();
            credentials.account_id = Some("second".into());
            state.accounts.insert(credentials);
            state.accounts.selected = "first".into();
        });
        let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt54, &state, cx));
        let second = model
            .with_account("second".into())
            .expect("account binding");
        cx.read(|cx| {
            assert!(
                model
                    .accounts(cx)
                    .iter()
                    .any(|account| account.id == "first" && account.selected)
            );
            assert!(
                second
                    .accounts(cx)
                    .iter()
                    .any(|account| account.id == "second" && account.selected)
            );
            assert_eq!(state.read(cx).accounts.selected, "first");
        });
        assert_eq!(second.account_id(), Some("second"));
        assert_eq!(model.account_id(), None);
    }

    #[gpui::test]
    async fn test_account_failover_signals_and_manual_policy(cx: &mut TestAppContext) {
        for (status, body, policy, expected) in [
            (
                401,
                "invalid_token",
                AccountSwitchPolicy::OnError,
                vec!["first", "refresh", "second"],
            ),
            (
                429,
                "rate_limit_exceeded",
                AccountSwitchPolicy::OnError,
                vec!["first", "second"],
            ),
            (
                402,
                "billing",
                AccountSwitchPolicy::OnError,
                vec!["first", "second"],
            ),
            (
                503,
                "overloaded",
                AccountSwitchPolicy::OnError,
                vec!["first", "first", "second"],
            ),
            (
                403,
                "insufficient_scope",
                AccountSwitchPolicy::OnError,
                vec!["first"],
            ),
            (
                403,
                r#"{"error":{"code":"insufficient_scope","required_scope":"responses.write"}}"#,
                AccountSwitchPolicy::OnError,
                vec!["first", "second"],
            ),
            (
                429,
                "rate_limit_exceeded",
                AccountSwitchPolicy::Manual,
                vec!["first"],
            ),
            (
                400,
                "invalid_request",
                AccountSwitchPolicy::OnError,
                vec!["first"],
            ),
        ] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let http = FakeHttpClient::create({
                let calls = calls.clone();
                move |request| {
                    let calls = calls.clone();
                    async move {
                        if request.uri().path() == "/oauth/token" {
                            calls.lock().push("refresh".to_owned());
                            return Ok(http_client::Response::builder()
                                .status(400)
                                .body(AsyncBody::from("invalid_grant"))?);
                        }
                        let account = request
                            .headers()
                            .get("chatgpt-account-id")
                            .expect("account header")
                            .to_str()?
                            .to_owned();
                        calls.lock().push(account.clone());
                        Ok(http_client::Response::builder()
                            .status(if account == "first" { status } else { 200 })
                            .body(AsyncBody::from(if account == "first" {
                                body
                            } else {
                                "data: [DONE]\n\n"
                            }))?)
                    }
                }
            });
            let mut credentials = make_fresh_credentials();
            credentials.account_id = Some("first".into());
            let state = make_state(http, Some(credentials), cx);
            state.update(cx, |state, _| {
                let mut second = make_fresh_credentials();
                second.account_id = Some("second".into());
                second.scopes = vec!["responses.write".into()];
                state.accounts.insert(second);
                state.accounts.selected = "first".into();
                state.configured_policy = Some(policy);
            });
            let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt54, &state, cx));
            let result = model
                .stream_completion(LanguageModelRequest::default(), &cx.to_async())
                .await;
            assert_eq!(
                result.is_ok(),
                expected.last() == Some(&"second"),
                "{status}: {body}"
            );
            drop(result);
            assert_eq!(*calls.lock(), expected, "{status}: {body}");
        }
    }

    #[gpui::test]
    async fn test_sse_quota_failure_uses_other_account(cx: &mut TestAppContext) {
        for partial_output in [false, true] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let http = FakeHttpClient::create({
                let calls = calls.clone();
                move |request| {
                    let calls = calls.clone();
                    async move {
                        let account = request
                            .headers()
                            .get("chatgpt-account-id")
                            .expect("account header")
                            .to_str()?
                            .to_owned();
                        calls.lock().push(account.clone());
                        let body = if account == "first" {
                            "data: {\"type\":\"response.created\",\"response\":{}}\n\ndata: {\"type\":\"error\",\"error\":{\"code\":\"usage_limit_reached\",\"message\":\"limit reached on your plan\"}}\n\n"
                        } else {
                            "data: [DONE]\n\n"
                        };
                        let body = if account == "first" && partial_output {
                            format!(
                                "data: {{\"type\":\"response.output_text.delta\",\"item_id\":\"message\",\"output_index\":0,\"delta\":\"Hello\"}}\n\n{body}"
                            )
                        } else {
                            body.to_owned()
                        };
                        Ok(http_client::Response::builder()
                            .status(200)
                            .body(AsyncBody::from(body))?)
                    }
                }
            });
            let mut credentials = make_fresh_credentials();
            credentials.account_id = Some("first".into());
            let state = make_state(http, Some(credentials), cx);
            state.update(cx, |state, _| {
                let mut second = make_fresh_credentials();
                second.account_id = Some("second".into());
                state.accounts.insert(second);
                state.accounts.selected = "first".into();
            });
            let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt54, &state, cx));
            let stream = model
                .stream_completion(LanguageModelRequest::default(), &cx.to_async())
                .await
                .expect("SSE failure should fail over");
            if partial_output {
                let events = stream.collect::<Vec<_>>().await;
                assert!(events.iter().any(Result::is_err));
                assert_eq!(*calls.lock(), vec!["first"], "do not replay partial output");
                drop(
                    model
                        .stream_completion(LanguageModelRequest::default(), &cx.to_async())
                        .await
                        .expect("next request uses other account"),
                );
            } else {
                drop(stream);
            }
            assert_eq!(*calls.lock(), vec!["first", "second"]);
            cx.read(|cx| {
                assert_eq!(
                    state
                        .read(cx)
                        .accounts
                        .accounts
                        .get("first")
                        .expect("first")
                        .quota_until_ms,
                    u64::MAX
                )
            });
        }
    }

    #[gpui::test]
    async fn test_accounts_use_separate_credential_records(cx: &mut TestAppContext) {
        let provider = Arc::new(FakeCredentialsProvider::new());
        let mut accounts = Accounts::default();
        for id in ["first", "second"] {
            let mut credentials = make_fresh_credentials();
            credentials.account_id = Some(id.into());
            credentials.access_token = "a".repeat(1800);
            accounts.insert(credentials);
        }
        let first = accounts.accounts.get_mut("first").expect("first");
        first.quota_until_ms = u64::MAX;
        first.status = "Plan quota reached".into();
        save_accounts(provider.as_ref(), &accounts, &cx.to_async())
            .await
            .expect("save");
        assert_eq!(provider.account_storage.lock().len(), 2);
        assert!(
            provider
                .account_storage
                .lock()
                .values()
                .all(|(_, bytes)| bytes.len() <= 2560)
        );
        let restored = load_accounts(provider.as_ref(), &cx.to_async())
            .await
            .expect("load");
        assert_eq!(restored.accounts.len(), 2);
        assert!(
            !restored
                .accounts
                .get("first")
                .expect("first")
                .available("gpt-5.4")
        );
        assert!(
            restored
                .accounts
                .get("second")
                .expect("second")
                .available("gpt-5.4")
        );
        let mut restored = restored;
        let credentials = restored
            .accounts
            .get("first")
            .expect("first")
            .credentials
            .clone()
            .expect("credentials");
        let first = restored.accounts.get_mut("first").expect("first");
        first.needs_reauth = true;
        first.status = "Sign in again".into();
        restored.insert(credentials);
        assert!(
            !restored
                .accounts
                .get("first")
                .expect("first")
                .available("gpt-5.4"),
            "reauth must not clear quota"
        );
        restored.accounts.remove("first");
        save_accounts(provider.as_ref(), &restored, &cx.to_async())
            .await
            .expect("remove");
        assert_eq!(provider.account_storage.lock().len(), 1);
    }

    #[gpui::test]
    async fn test_unauthorized_refreshes_same_account_before_failover(cx: &mut TestAppContext) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let http = FakeHttpClient::create({
            let calls = calls.clone();
            move |request| {
                let calls = calls.clone();
                async move {
                    if request.uri().path() == "/oauth/token" {
                        calls.lock().push("refresh".to_owned());
                        return Ok(http_client::Response::builder()
                            .status(200)
                            .body(AsyncBody::from(fake_token_response()))?);
                    }
                    let account = request
                        .headers()
                        .get("chatgpt-account-id")
                        .expect("account header")
                        .to_str()?
                        .to_owned();
                    calls.lock().push(account.clone());
                    Ok(http_client::Response::builder()
                        .status(if account == "first" { 401 } else { 200 })
                        .body(AsyncBody::from(if account == "first" {
                            "invalid_token"
                        } else {
                            "data: [DONE]\n\n"
                        }))?)
                }
            }
        });
        let mut credentials = make_fresh_credentials();
        credentials.account_id = Some("first".into());
        let state = make_state(http, Some(credentials), cx);
        state.update(cx, |state, _| {
            let mut second = make_fresh_credentials();
            second.account_id = Some("second".into());
            state.accounts.insert(second);
            state.accounts.selected = "first".into();
        });
        let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt54, &state, cx));
        let stream = model
            .stream_completion(LanguageModelRequest::default(), &cx.to_async())
            .await
            .expect("failover succeeds");
        drop(stream);
        assert_eq!(*calls.lock(), vec!["first", "refresh", "first", "second"]);
        cx.read(|cx| {
            assert!(
                state
                    .read(cx)
                    .accounts
                    .accounts
                    .get("first")
                    .expect("first")
                    .needs_reauth
            )
        });
    }

    #[gpui::test]
    async fn test_plan_quota_freezes_account_across_requests(cx: &mut TestAppContext) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let http = FakeHttpClient::create({
            let calls = calls.clone();
            move |request| {
                let calls = calls.clone();
                async move {
                    let account = request
                        .headers()
                        .get("chatgpt-account-id")
                        .expect("account header")
                        .to_str()?
                        .to_owned();
                    calls.lock().push(account.clone());
                    Ok(http_client::Response::builder()
                        .status(if account == "first" { 429 } else { 200 })
                        .body(AsyncBody::from(if account == "first" {
                            "limit reached on your plan"
                        } else {
                            "data: [DONE]\n\n"
                        }))?)
                }
            }
        });
        let mut credentials = make_fresh_credentials();
        credentials.account_id = Some("first".into());
        let state = make_state(http, Some(credentials), cx);
        state.update(cx, |state, _| {
            let mut second = make_fresh_credentials();
            second.account_id = Some("second".into());
            state.accounts.insert(second);
            state.accounts.selected = "first".into();
        });
        let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt54, &state, cx));
        for _ in 0..2 {
            let stream = model
                .stream_completion(LanguageModelRequest::default(), &cx.to_async())
                .await
                .expect("failover succeeds");
            drop(stream);
        }
        assert_eq!(*calls.lock(), vec!["first", "second", "second"]);
        cx.read(|cx| {
            assert_eq!(
                state
                    .read(cx)
                    .accounts
                    .accounts
                    .get("first")
                    .expect("first")
                    .quota_until_ms,
                u64::MAX
            )
        });
    }

    #[gpui::test]
    async fn test_concurrent_refresh_deduplicates(cx: &mut TestAppContext) {
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let refresh_count_clone = refresh_count.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let refresh_count = refresh_count_clone.clone();
            async move {
                refresh_count.fetch_add(1, Ordering::SeqCst);
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        // Spawn two concurrent refresh attempts.
        let weak1 = weak_state.clone();
        let http1 = http.clone();
        let task1 =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak1, &http1, &mut cx).await);

        let weak2 = weak_state.clone();
        let http2 = http.clone();
        let task2 =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak2, &http2, &mut cx).await);

        // Drive both to completion.
        cx.run_until_parked();
        let result1 = task1.await;
        let result2 = task2.await;

        assert!(result1.is_ok(), "first refresh should succeed");
        assert!(result2.is_ok(), "second refresh should succeed");
        assert_eq!(result1.unwrap().access_token, "fresh_access");
        assert_eq!(result2.unwrap().access_token, "fresh_access");
        assert_eq!(
            refresh_count.load(Ordering::SeqCst),
            1,
            "refresh_token should only be called once despite two concurrent callers"
        );
    }

    #[gpui::test]
    async fn test_fresh_credentials_skip_refresh(cx: &mut TestAppContext) {
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let refresh_count_clone = refresh_count.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let refresh_count = refresh_count_clone.clone();
            async move {
                refresh_count.fetch_add(1, Ordering::SeqCst);
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_fresh_credentials()), cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().access_token, "fresh_access");
        assert_eq!(
            refresh_count.load(Ordering::SeqCst),
            0,
            "no refresh should happen when credentials are fresh"
        );
    }

    #[gpui::test]
    async fn test_no_credentials_returns_no_api_key(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), None, cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        assert!(matches!(
            result,
            Err(LanguageModelCompletionError::NoApiKey { .. })
        ));
    }

    #[gpui::test]
    async fn test_fatal_refresh_clears_auth_state(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(401)
                .body(http_client::AsyncBody::from(r#"{"error":"invalid_grant"}"#))?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        cx.run_until_parked();

        assert!(result.is_err(), "fatal refresh should return an error");
        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                !s.is_authenticated(),
                "credentials should be cleared on fatal refresh failure"
            );
            assert!(
                s.last_auth_error.is_some(),
                "last_auth_error should be set on fatal refresh failure"
            );
        });
    }

    #[gpui::test]
    async fn test_transient_refresh_keeps_credentials(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(500)
                .body(http_client::AsyncBody::from("Internal Server Error"))?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        cx.run_until_parked();

        assert!(result.is_err(), "transient refresh should return an error");
        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                s.is_authenticated(),
                "credentials should be kept on transient refresh failure"
            );
            assert!(
                s.last_auth_error.is_none(),
                "last_auth_error should not be set on transient refresh failure"
            );
        });
    }

    #[gpui::test]
    async fn test_cancel_sign_in_drops_pending_task(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, None, cx);
        let (continue_sign_in_tx, continue_sign_in_rx) = futures::channel::oneshot::channel::<()>();

        state.update(cx, |state, cx| {
            let task = cx.spawn(async move |_this, _cx| {
                continue_sign_in_rx.await?;
                anyhow::Ok(())
            });
            state.sign_in_state = SignInState::Authorizing(task);
        });

        cx.read(|cx| assert!(state.read(cx).is_signing_in()));
        state.update(cx, |state, cx| state.cancel_sign_in(cx));
        cx.run_until_parked();
        cx.read(|cx| assert!(!state.read(cx).is_signing_in()));
        assert!(
            continue_sign_in_tx.send(()).is_err(),
            "canceling sign-in should drop the task"
        );
    }

    #[gpui::test]
    async fn test_sign_in_task_remains_alive_while_persisting_credentials(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, None, cx);
        let (begin_persisting_tx, begin_persisting_rx) = futures::channel::oneshot::channel::<()>();
        let (finish_persisting_tx, finish_persisting_rx) =
            futures::channel::oneshot::channel::<()>();

        state.update(cx, |state, cx| {
            let task = cx.spawn(async move |this, cx| {
                begin_persisting_rx.await?;
                this.update(cx, |state, cx| {
                    state.begin_persisting_credentials(cx);
                })?;
                finish_persisting_rx.await?;
                this.update(cx, |state, cx| {
                    state.sign_in_state = SignInState::Idle;
                    cx.notify();
                })?;
                anyhow::Ok(())
            });
            state.sign_in_state = SignInState::Authorizing(task);
        });

        begin_persisting_tx
            .send(())
            .expect("sign-in task should be waiting to persist credentials");
        cx.run_until_parked();
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.is_signing_in());
            assert!(!state.is_sign_in_cancellable());
        });

        state.update(cx, |state, cx| state.cancel_sign_in(cx));
        cx.run_until_parked();
        cx.read(|cx| assert!(state.read(cx).is_signing_in()));

        finish_persisting_tx
            .send(())
            .expect("sign-in task should still be persisting credentials");
        cx.run_until_parked();
        cx.read(|cx| assert!(!state.read(cx).is_signing_in()));
    }

    #[gpui::test]
    async fn test_sign_out_during_refresh_discards_result(cx: &mut TestAppContext) {
        let (gate_tx, gate_rx) = futures::channel::oneshot::channel::<()>();
        let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
        let gate_rx_clone = gate_rx.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let gate_rx = gate_rx_clone.clone();
            async move {
                // Wait until the gate is opened, simulating a slow network.
                let rx = gate_rx.lock().take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        // Start a refresh
        let weak = weak_state.clone();
        let http_clone = http.clone();
        let refresh_task =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await);

        cx.run_until_parked();

        // Sign out while the refresh is in-flight
        state.update(cx, |state, cx| {
            state.sign_out(cx).detach();
        });
        cx.run_until_parked();

        // Now let the refresh respond by opening the gate
        let _ = gate_tx.send(());
        cx.run_until_parked();

        let result = refresh_task.await;
        assert!(result.is_err(), "refresh should fail after sign-out");

        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                !s.is_authenticated(),
                "sign-out should have cleared credentials"
            );
        });
    }

    #[gpui::test]
    async fn test_sign_out_completes_fully(cx: &mut TestAppContext) {
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        // Pre-populate the credential store
        creds_provider
            .storage
            .lock()
            .replace(("Bearer".to_string(), b"some-creds".to_vec()));

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state_with_credentials_provider(
            http,
            Some(make_fresh_credentials()),
            creds_provider.clone(),
            cx,
        );

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));

        cx.run_until_parked();
        sign_out_task.await.expect("sign-out should succeed");

        assert!(
            creds_provider.storage.lock().is_none(),
            "credential store should be empty after sign-out"
        );
        cx.read(|cx| {
            assert!(
                !state.read(cx).is_authenticated(),
                "state should show not authenticated"
            );
        });
    }

    #[gpui::test]
    async fn test_initial_load_restores_persisted_credentials(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_json = serde_json::to_vec(&creds).unwrap();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider
            .storage
            .lock()
            .replace(("Bearer".to_string(), creds_json));

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|request| async move {
            assert_eq!(
                request.uri().to_string(),
                "https://chatgpt.com/backend-api/codex/models?client_version=0.0.0"
            );
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(
                    serde_json::json!({
                        "models": [{
                            "slug": "gpt-account-default",
                            "display_name": "Account Default",
                            "default_reasoning_level": "medium",
                            "supported_reasoning_levels": [{
                                "effort": "medium",
                                "description": "Medium"
                            }],
                            "visibility": "list",
                            "priority": 0,
                            "additional_speed_tiers": [],
                            "service_tiers": [],
                            "context_window": 128_000,
                            "max_context_window": null,
                            "input_modalities": ["text"]
                        }]
                    })
                    .to_string(),
                ))?)
        });

        let state = cx.new(|cx| State::new(http, creds_provider, cx));

        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");

        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.is_authenticated());
            assert!(state.load_task().is_none());
            assert_eq!(
                state.available_models().first().map(ChatGptModel::id),
                Some("gpt-account-default")
            );
        });
    }

    #[gpui::test]
    async fn test_model_catalog_uses_account_visible_models(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(|request| async move {
            assert_eq!(request.method(), Method::GET);
            assert_eq!(
                request.uri().to_string(),
                "https://chatgpt.com/backend-api/codex/models?client_version=1.2.3"
            );
            assert_eq!(
                request
                    .headers()
                    .get("authorization")
                    .and_then(|value| value.to_str().ok()),
                Some("Bearer fresh_access")
            );
            assert_eq!(
                request
                    .headers()
                    .get("chatgpt-account-id")
                    .and_then(|value| value.to_str().ok()),
                Some("account-123")
            );
            assert_eq!(
                request
                    .headers()
                    .get("originator")
                    .and_then(|value| value.to_str().ok()),
                Some("zed")
            );
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(
                    serde_json::json!({
                        "models": [
                            {
                                "slug": "gpt-5.6-sol",
                                "display_name": "GPT-5.6 Sol",
                                "default_reasoning_level": "low",
                                "supported_reasoning_levels": [{
                                    "effort": "low",
                                    "description": "Low"
                                }],
                                "visibility": "hide",
                                "priority": 0,
                                "additional_speed_tiers": ["fast"],
                                "service_tiers": [],
                                "context_window": 372_000,
                                "max_context_window": null,
                                "input_modalities": ["text", "image"]
                            },
                            {
                                "slug": "gpt-5.6-luna",
                                "display_name": "GPT-5.6 Luna",
                                "default_reasoning_level": "medium",
                                "supported_reasoning_levels": [{
                                    "effort": "medium",
                                    "description": "Medium"
                                }],
                                "visibility": "list",
                                "priority": 2,
                                "additional_speed_tiers": [],
                                "service_tiers": [{"id": "priority"}],
                                "context_window": 372_000,
                                "max_context_window": null,
                                "input_modalities": ["text", "image"]
                            },
                            {
                                "slug": "gpt-5.5",
                                "display_name": "GPT-5.5",
                                "default_reasoning_level": "medium",
                                "supported_reasoning_levels": [{
                                    "effort": "medium",
                                    "description": "Medium"
                                }],
                                "visibility": "list",
                                "priority": 1,
                                "additional_speed_tiers": [],
                                "service_tiers": [],
                                "context_window": 272_000,
                                "max_context_window": null,
                                "input_modalities": ["text"]
                            }
                        ]
                    })
                    .to_string(),
                ))?)
        });
        let mut credentials = make_fresh_credentials();
        credentials.account_id = Some("account-123".to_string());
        let state = make_state(http_client, Some(credentials), cx);
        state.update(cx, |state, _cx| {
            state.client_version = "1.2.3".into();
        });

        state
            .update(cx, |state, cx| state.refresh_model_catalog(cx))
            .await
            .expect("model discovery should succeed");

        cx.read(|cx| {
            let state = state.read(cx);
            let model_ids = state
                .available_models()
                .iter()
                .map(ChatGptModel::id)
                .collect::<Vec<_>>();
            assert_eq!(model_ids, ["gpt-5.5", "gpt-5.6-luna"]);
            assert_eq!(
                state.default_model().as_ref().map(ChatGptModel::id),
                Some("gpt-5.5")
            );
            assert_eq!(
                state.default_fast_model().as_ref().map(ChatGptModel::id),
                Some("gpt-5.6-luna")
            );
            let default_model = state
                .available_models()
                .iter()
                .find(|model| model.id() == "gpt-5.5")
                .expect("default model should be present");
            let fast_model = state
                .available_models()
                .iter()
                .find(|model| model.id() == "gpt-5.6-luna")
                .expect("fast model should be present");
            assert!(!default_model.supports_images());
            assert!(fast_model.supports_priority());
            assert!(state.model_catalog_error().is_none());
        });
    }

    #[gpui::test]
    async fn test_model_catalog_failure_preserves_fallback_models(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(|_| async move {
            Ok(http_client::Response::builder()
                .status(500)
                .body(http_client::AsyncBody::from("backend unavailable"))?)
        });
        let state = make_state(http_client, Some(make_fresh_credentials()), cx);
        let fallback_model_ids = cx.read(|cx| {
            state
                .read(cx)
                .available_models()
                .iter()
                .map(ChatGptModel::id)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });

        state
            .update(cx, |state, cx| state.refresh_model_catalog(cx))
            .await
            .expect_err("model discovery should fail");

        cx.read(|cx| {
            let state = state.read(cx);
            let model_ids = state
                .available_models()
                .iter()
                .map(ChatGptModel::id)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            assert_eq!(model_ids, fallback_model_ids);
            assert!(
                state
                    .model_catalog_error()
                    .is_some_and(|error| error.contains("backend unavailable"))
            );
        });
    }

    #[gpui::test]
    async fn test_model_catalog_request_times_out(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(|_| {
            futures::future::pending::<Result<http_client::Response<AsyncBody>>>()
        });
        let state = make_state(http_client, Some(make_fresh_credentials()), cx);

        let refresh_task = state.update(cx, |state, cx| state.refresh_model_catalog(cx));
        cx.run_until_parked();
        cx.executor().advance_clock(MODEL_CATALOG_REQUEST_TIMEOUT);
        cx.run_until_parked();

        let error = refresh_task
            .await
            .expect_err("model discovery should time out");
        assert!(
            error.to_string().contains("timed out after 5s"),
            "unexpected model discovery error: {error:#}"
        );
    }

    #[gpui::test]
    async fn test_obsolete_model_catalog_cannot_replace_newer_models(cx: &mut TestAppContext) {
        let (release_obsolete_request, obsolete_request) =
            futures::channel::oneshot::channel::<()>();
        let obsolete_request = Arc::new(Mutex::new(Some(obsolete_request)));
        let request_count = Arc::new(AtomicUsize::new(0));
        let http_client = FakeHttpClient::create({
            let obsolete_request = obsolete_request.clone();
            let request_count = request_count.clone();
            move |_| {
                let obsolete_request = obsolete_request.clone();
                let request_count = request_count.clone();
                async move {
                    let request_index = request_count.fetch_add(1, Ordering::SeqCst);
                    if request_index == 0 {
                        let receiver = obsolete_request.lock().take();
                        if let Some(receiver) = receiver {
                            receiver.await.expect("obsolete request should be released");
                        }
                    }
                    let model_id = if request_index == 0 {
                        "obsolete-model"
                    } else {
                        "current-model"
                    };
                    Ok(http_client::Response::builder().status(200).body(
                        http_client::AsyncBody::from(
                            serde_json::json!({
                                "models": [{
                                    "slug": model_id,
                                    "display_name": model_id,
                                    "default_reasoning_level": "medium",
                                    "supported_reasoning_levels": [],
                                    "visibility": "list",
                                    "priority": 0,
                                    "additional_speed_tiers": [],
                                    "service_tiers": [],
                                    "context_window": 128_000,
                                    "max_context_window": null,
                                    "input_modalities": ["text"]
                                }]
                            })
                            .to_string(),
                        ),
                    )?)
                }
            }
        });
        let state = make_state(http_client, Some(make_fresh_credentials()), cx);

        let obsolete_refresh = state.update(cx, |state, cx| state.refresh_model_catalog(cx));
        cx.run_until_parked();
        state
            .update(cx, |state, cx| state.refresh_model_catalog(cx))
            .await
            .expect("newer model discovery should succeed");
        release_obsolete_request
            .send(())
            .expect("obsolete request should remain connected");
        obsolete_refresh
            .await
            .expect("obsolete model discovery may still complete");

        cx.read(|cx| {
            assert_eq!(
                state
                    .read(cx)
                    .available_models()
                    .first()
                    .map(ChatGptModel::id),
                Some("current-model")
            );
        });
    }

    #[gpui::test]
    async fn test_server_side_compaction_streams_from_codex_responses(cx: &mut TestAppContext) {
        let compaction_request_count = Arc::new(AtomicUsize::new(0));
        let http_client = FakeHttpClient::create({
            let compaction_request_count = compaction_request_count.clone();
            move |request| {
                let compaction_request_count = compaction_request_count.clone();
                async move {
                    assert_eq!(
                        request.uri().to_string(),
                        "https://chatgpt.com/backend-api/codex/responses"
                    );
                    assert_eq!(
                        request
                            .headers()
                            .get("authorization")
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer fresh_access")
                    );
                    assert_eq!(
                        request
                            .headers()
                            .get("chatgpt-account-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("account-123")
                    );
                    assert_eq!(
                        request
                            .headers()
                            .get("session-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("thread-123")
                    );
                    assert_eq!(
                        request
                            .headers()
                            .get("thread-id")
                            .and_then(|value| value.to_str().ok()),
                        Some("thread-123")
                    );
                    let mut request_body = String::new();
                    smol::io::AsyncReadExt::read_to_string(
                        &mut request.into_body(),
                        &mut request_body,
                    )
                    .await?;
                    let request_body: serde_json::Value = serde_json::from_str(&request_body)?;
                    assert_eq!(
                        request_body["context_management"],
                        serde_json::json!([{
                            "type": "compaction",
                            "compact_threshold": 100_000,
                        }])
                    );
                    compaction_request_count.fetch_add(1, Ordering::SeqCst);
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(http_client::AsyncBody::from(compaction_response_stream()))?)
                }
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let mut credentials = make_fresh_credentials();
        credentials.account_id = Some("account-123".to_string());
        let state = make_state(http, Some(credentials), cx);
        let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt55, &state, cx));
        assert!(model.supports_server_side_compaction());
        assert!(model.supports_explicit_compaction());

        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec![language_model::MessageContent::Text("Hello".into())],
                cache: false,
                reasoning_details: None,
            }],
            compact_at_tokens: Some(100_000),
            thread_id: Some("thread-123".to_string()),
            ..Default::default()
        };
        let async_cx = cx.to_async();
        let events = model
            .stream_completion(request, &async_cx)
            .await
            .expect("the response stream should start")
            .collect::<Vec<_>>()
            .await;

        assert_eq!(compaction_request_count.load(Ordering::SeqCst), 1);
        assert!(matches!(
            events.first(),
            Some(Ok(LanguageModelCompletionEvent::Compaction(
                language_model::CompactionUpdate::Started
            )))
        ));
        let Some(Ok(LanguageModelCompletionEvent::Compaction(
            language_model::CompactionUpdate::Finished(
                language_model::CompactedContext::ProviderState(compaction_state),
            ),
        ))) = events.get(1)
        else {
            panic!("expected the streamed provider compaction state");
        };
        assert_eq!(compaction_state.provider_id(), &PROVIDER_ID);
        let items = open_ai::responses::provider_compaction_items(&compaction_state, &PROVIDER_ID)
            .expect("the compacted state should parse")
            .expect("the compacted state should be owned by the subscription provider");
        assert_eq!(
            items,
            vec![serde_json::json!({
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "opaque-state",
            })]
        );
    }

    #[gpui::test]
    async fn test_explicit_compaction_streams_with_codex_compaction_trigger(
        cx: &mut TestAppContext,
    ) {
        let http_client = FakeHttpClient::create(move |request| async move {
            assert_eq!(
                request.uri().to_string(),
                "https://chatgpt.com/backend-api/codex/responses"
            );
            assert_eq!(
                request
                    .headers()
                    .get("session-id")
                    .and_then(|value| value.to_str().ok()),
                Some("thread-123")
            );
            assert_eq!(
                request
                    .headers()
                    .get("thread-id")
                    .and_then(|value| value.to_str().ok()),
                Some("thread-123")
            );
            let mut request_body = String::new();
            smol::io::AsyncReadExt::read_to_string(&mut request.into_body(), &mut request_body)
                .await?;
            let request_body: serde_json::Value = serde_json::from_str(&request_body)?;
            assert!(request_body.get("context_management").is_none());
            assert_eq!(
                request_body["input"]
                    .as_array()
                    .and_then(|input| input.last()),
                Some(&serde_json::json!({"type": "compaction_trigger"}))
            );
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(compaction_response_stream()))?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http, Some(make_fresh_credentials()), cx);
        let model = cx.read(|cx| create_language_model(ChatGptModel::Gpt55, &state, cx));
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec![language_model::MessageContent::Text("Hello".into())],
                cache: false,
                reasoning_details: None,
            }],
            compact_at_tokens: Some(100_000),
            thread_id: Some("thread-123".to_string()),
            ..Default::default()
        };

        let result = model
            .compact(request, &cx.to_async())
            .await
            .expect("manual compaction should succeed");
        let language_model::CompactedContext::ProviderState(compaction_state) = result.context
        else {
            panic!("expected provider compaction state");
        };
        let items = open_ai::responses::provider_compaction_items(&compaction_state, &PROVIDER_ID)
            .expect("the compacted state should parse")
            .expect("the compacted state should be owned by the subscription provider");
        assert_eq!(
            items,
            vec![serde_json::json!({
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "opaque-state",
            })]
        );
    }

    fn compaction_response_stream() -> String {
        let compaction_item = serde_json::json!({
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": "opaque-state",
        });
        [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": compaction_item,
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": compaction_item,
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [],
                },
            }),
        ]
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
    }

    struct FakeCredentialsProvider {
        storage: Mutex<Option<(String, Vec<u8>)>>,
        account_storage: Mutex<BTreeMap<String, (String, Vec<u8>)>>,
    }

    impl FakeCredentialsProvider {
        fn new() -> Self {
            Self {
                storage: Mutex::new(None),
                account_storage: Mutex::new(BTreeMap::new()),
            }
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async move {
                Ok(if url == CREDENTIALS_KEY {
                    self.storage.lock().clone()
                } else {
                    self.account_storage.lock().get(url).cloned()
                })
            })
        }

        fn write_credentials<'a>(
            &'a self,
            url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            if url == CREDENTIALS_KEY {
                self.storage
                    .lock()
                    .replace((username.to_owned(), password.to_vec()));
            } else {
                self.account_storage
                    .lock()
                    .insert(url.to_owned(), (username.to_owned(), password.to_vec()));
            }
            Box::pin(async { Ok(()) })
        }

        fn delete_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            if url == CREDENTIALS_KEY {
                *self.storage.lock() = None;
            } else {
                self.account_storage.lock().remove(url);
            }
            Box::pin(async { Ok(()) })
        }
    }

    fn make_state(
        http_client: Arc<dyn HttpClient>,
        credentials: Option<CodexCredentials>,
        cx: &mut TestAppContext,
    ) -> Entity<State> {
        make_state_with_credentials_provider(
            http_client,
            credentials,
            Arc::new(FakeCredentialsProvider::new()),
            cx,
        )
    }

    fn make_state_with_credentials_provider(
        http_client: Arc<dyn HttpClient>,
        credentials: Option<CodexCredentials>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut TestAppContext,
    ) -> Entity<State> {
        cx.new(|_cx| State {
            accounts: {
                let mut accounts = Accounts::default();
                if let Some(credentials) = credentials {
                    accounts.insert(credentials);
                }
                accounts
            },
            usage_requests: BTreeMap::new(),
            configured_policy: None,
            persistence_task: None,
            sign_in_state: SignInState::Idle,
            refresh_tasks: BTreeMap::new(),
            load_task: None,
            credentials_provider,
            http_client,
            client_version: "0.0.0".into(),
            available_models: ChatGptModel::all(),
            auth_generation: 0,
            model_catalog_generation: 0,
            last_auth_error: None,
            last_model_catalog_error: None,
        })
    }

    fn make_expired_credentials() -> CodexCredentials {
        CodexCredentials {
            access_token: "old_access".to_string(),
            refresh_token: "old_refresh".to_string(),
            scopes: Vec::new(),
            expires_at_ms: 0,
            account_id: None,
            email: None,
        }
    }

    fn make_fresh_credentials() -> CodexCredentials {
        CodexCredentials {
            access_token: "fresh_access".to_string(),
            refresh_token: "fresh_refresh".to_string(),
            scopes: Vec::new(),
            expires_at_ms: now_ms() + 3_600_000,
            account_id: None,
            email: None,
        }
    }

    fn fake_token_response() -> String {
        serde_json::json!({
            "access_token": "fresh_access",
            "refresh_token": "fresh_refresh",
            "expires_in": 3600
        })
        .to_string()
    }
}
