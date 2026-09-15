use crate::AllLanguageModelSettings;
use anyhow::{Result, anyhow};
use credentials_provider::CredentialsProvider;
use futures::future::Shared;
use gpui::{App, Context, Entity, SharedString, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, FastModeConfirmation, IconOrSvg, InlineDescription, LanguageModel,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, ProviderSettingsView,
};
use openai_subscribed::{
    AccountSwitchPolicy, PROVIDER_ID, PROVIDER_NAME, State, create_language_model,
};
use settings::{Settings as _, SettingsStore};
use std::sync::Arc;
use ui::{ConfiguredApiCard, prelude::*};

const SUBSCRIPTION_DESCRIPTION: &str =
    "Sign in with your ChatGPT Plus or Pro subscription to use OpenAI models in Zed's agent.";

pub struct OpenAiSubscribedProvider {
    state: Entity<State>,
}

impl OpenAiSubscribedProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let apply_policy = |state: &mut State, cx: &mut Context<State>| {
            let policy = AllLanguageModelSettings::try_get(cx)
                .and_then(|settings| settings.openai_account_switch_policy)
                .map(|policy| match policy {
                    settings::OpenAiAccountSwitchPolicy::Manual => AccountSwitchPolicy::Manual,
                    settings::OpenAiAccountSwitchPolicy::OnError => AccountSwitchPolicy::OnError,
                });
            state.configure_switch_policy(policy, cx);
        };
        let state = cx.new(|cx| {
            let mut state = State::new(http_client, credentials_provider, cx);
            apply_policy(&mut state, cx);
            cx.observe_global::<SettingsStore>(apply_policy).detach();
            state
        });
        Self { state }
    }
}

impl LanguageModelProviderState for OpenAiSubscribedProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiSubscribedProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAiGptSub)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .default_model()
            .map(|model| create_language_model(model, &self.state, cx))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .default_fast_model()
            .map(|model| create_language_model(model, &self.state, cx))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .available_models()
            .iter()
            .cloned()
            .map(|model| create_language_model(model, &self.state, cx))
            .collect()
    }

    fn recommended_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.default_model(cx).into_iter().collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated(cx) {
            return Task::ready(Ok(()));
        }
        let load_task: Option<Shared<_>> = self.state.read(cx).load_task();
        if let Some(load_task) = load_task {
            let weak_state = self.state.downgrade();
            cx.spawn(async move |cx| {
                load_task
                    .await
                    .map_err(|error| AuthenticateError::Other(anyhow!("{error:#}")))?;
                let is_auth = weak_state
                    .read_with(&*cx, |state, _| state.is_authenticated())
                    .map_err(AuthenticateError::Other)?;
                if is_auth {
                    Ok(())
                } else {
                    Err(AuthenticateError::CredentialsNotFound)
                }
            })
        } else {
            Task::ready(Err(AuthenticateError::CredentialsNotFound))
        }
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let title = if is_authenticated {
            None
        } else {
            Some("Configure ChatGPT".into())
        };
        let description = if is_authenticated {
            None
        } else {
            Some(InlineDescription::Text(SUBSCRIPTION_DESCRIPTION.into()))
        };

        Some(ProviderSettingsView::Inline(
            language_model::InlineProviderSettings {
                title,
                description,
                create_view: Arc::new({
                    let state = self.state.clone();
                    move |_window, cx| {
                        cx.new(|cx| ConfigurationView {
                            _subscription: cx.observe(&state, |_, _, cx| cx.notify()),
                            state: state.clone(),
                            compact: true,
                        })
                        .into()
                    }
                }),
            },
        ))
    }

    fn authentication_error_message(&self) -> SharedString {
        "Your ChatGPT subscription session is invalid or has expired. \
        Sign in again via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn missing_credentials_error_message(&self) -> SharedString {
        "You are not signed in to your ChatGPT account. \
        Sign in via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn fast_mode_confirmation(&self, _cx: &App) -> Option<FastModeConfirmation> {
        Some(FastModeConfirmation {
            title: "Enable Fast Mode for OpenAI?".into(),
            message: "Fast mode sends requests using OpenAI's Priority processing tier, which \
                targets significantly lower latency than the standard tier and is billed at a \
                premium per-token rate."
                .into(),
        })
    }
}

struct ConfigurationView {
    _subscription: gpui::Subscription,
    state: Entity<State>,
    /// When `true`, the description is rendered elsewhere (the settings row's
    /// left column), so it's omitted here to avoid duplication.
    compact: bool,
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);

        let accounts = state.accounts();
        let policy = state.switch_policy();
        let policy_state = self.state.clone();
        let account_state = self.state.clone();
        let has_accounts = !accounts.is_empty();
        let last_auth_error = state.last_auth_error();
        let model_catalog_error = state.model_catalog_error();
        let provider_state = self.state.clone();
        let cancel_provider_state = self.state.clone();

        let is_signing_in = state.is_signing_in();
        let is_sign_in_cancellable = state.is_sign_in_cancellable();
        let button_label = if is_signing_in {
            "Signing in…"
        } else {
            if has_accounts {
                "Add Account"
            } else {
                "Sign In"
            }
        };

        v_flex()
            .gap_2()
            .children(accounts.into_iter().map(|account| {
                let reset_state = account_state.clone();
                let reauth_state = account_state.clone();
                let reset_id = account.id.clone();
                let remove_state = account_state.clone();
                let select_state = account_state.clone();
                let remove_id = account.id.clone();
                let select_id = account.id.clone();
                v_flex()
                    .gap_1()
                    .child(
                        ConfiguredApiCard::new(
                            SharedString::from(format!("account-{}", account.id)),
                            SharedString::from(account.label),
                        )
                        .button_label("Remove")
                        .on_click(move |_, _, cx| {
                            remove_state
                                .update(cx, |state, cx| state.remove_account(&remove_id, cx));
                        }),
                    )
                    .child(
                        Button::new(
                            SharedString::from(format!("select-{}", account.id)),
                            if account.selected {
                                "Default account"
                            } else {
                                "Use by default"
                            },
                        )
                        .disabled(account.selected)
                        .on_click(move |_, _, cx| {
                            select_state.update(cx, |state, cx| {
                                state.select_account(select_id.clone(), cx)
                            });
                        }),
                    )
                    .when(!account.status.is_empty(), |this| {
                        this.child(Label::new(account.status.clone()).color(Color::Warning))
                            .child(
                                Button::new(
                                    SharedString::from(format!("reauth-{}", account.id)),
                                    "Sign in again",
                                )
                                .disabled(is_signing_in)
                                .on_click(move |_, _, cx| {
                                    reauth_state.update(cx, |state, cx| state.sign_in(cx));
                                }),
                            )
                            .when(account.status == "Plan quota reached", |this| {
                                this.child(
                                    Button::new(
                                        SharedString::from(format!("reset-{}", account.id)),
                                        "I've updated my quota",
                                    )
                                    .on_click(
                                        move |_, _, cx| {
                                            reset_state.update(cx, |state, cx| {
                                                state.reset_account_availability(&reset_id, cx)
                                            });
                                        },
                                    ),
                                )
                            })
                    })
            }))
            .when(has_accounts, |this| {
                this.child(
                    Button::new(
                        "account-switch-policy",
                        match policy {
                            AccountSwitchPolicy::Manual => "Account switching: Manual",
                            AccountSwitchPolicy::OnError => "Account switching: On error",
                        },
                    )
                    .on_click(move |_, _, cx| {
                        let policy = match policy {
                            AccountSwitchPolicy::Manual => AccountSwitchPolicy::OnError,
                            AccountSwitchPolicy::OnError => AccountSwitchPolicy::Manual,
                        };
                        policy_state.update(cx, |state, cx| {
                            state.configure_switch_policy(Some(policy), cx)
                        });
                        let fs = <dyn fs::Fs>::global(cx);
                        settings::update_settings_file(fs, cx, move |settings, _| {
                            settings
                                .language_models
                                .get_or_insert_default()
                                .openai_subscribed
                                .get_or_insert_default()
                                .account_switch_policy = Some(match policy {
                                AccountSwitchPolicy::Manual => {
                                    settings::OpenAiAccountSwitchPolicy::Manual
                                }
                                AccountSwitchPolicy::OnError => {
                                    settings::OpenAiAccountSwitchPolicy::OnError
                                }
                            });
                        });
                    }),
                )
            })
            .when(!self.compact, |this| {
                this.child(Label::new(SUBSCRIPTION_DESCRIPTION))
            })
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("sign-in", button_label)
                            .when(!self.compact, |this| this.full_width())
                            .style(ButtonStyle::Outlined)
                            .size(ButtonSize::Medium)
                            .loading(is_signing_in)
                            .disabled(is_signing_in)
                            .on_click(move |_, _window, cx| {
                                provider_state.update(cx, |state, cx| state.sign_in(cx));
                            }),
                    )
                    .when(is_sign_in_cancellable, |this| {
                        this.child(
                            Button::new("cancel-sign-in", "Cancel")
                                .style(ButtonStyle::Subtle)
                                .size(ButtonSize::Medium)
                                .on_click(move |_, _window, cx| {
                                    cancel_provider_state
                                        .update(cx, |state, cx| state.cancel_sign_in(cx));
                                }),
                        )
                    }),
            )
            .when_some(model_catalog_error, |this, error| {
                this.child(Label::new(error).color(Color::Warning))
            })
            .when_some(last_auth_error, |this, error| {
                this.child(
                    h_flex()
                        .gap_1()
                        .justify_center()
                        .child(
                            Icon::new(IconName::XCircle)
                                .color(Color::Error)
                                .size(IconSize::Small),
                        )
                        .child(Label::new(error).color(Color::Muted)),
                )
            })
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AsyncApp, TestAppContext};
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::future::Future;
    use std::pin::Pin;

    #[gpui::test]
    async fn test_authenticate_awaits_initial_load(cx: &mut TestAppContext) {
        let creds_json = serde_json::json!({
            "access_token": "fresh_access",
            "refresh_token": "fresh_refresh",
            "expires_at_ms": u64::MAX,
            "account_id": null,
            "email": null,
        });
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.storage.lock().replace((
            "Bearer".to_string(),
            serde_json::to_vec(&creds_json).unwrap(),
        ));

        let http_client = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(
                    serde_json::json!({
                        "models": [{
                            "slug": "gpt-account-default",
                            "display_name": "Account Default",
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
                ))?)
        });

        let provider =
            cx.update(|cx| OpenAiSubscribedProvider::new(http_client, creds_provider, cx));

        // Before load completes, authenticate should still await the load.
        let auth_task = cx.update(|cx| provider.authenticate(cx));

        // Drive the load to completion.
        cx.run_until_parked();

        let result = auth_task.await;
        assert!(
            result.is_ok(),
            "authenticate should succeed after load completes with valid credentials"
        );
        cx.update(|cx| {
            let models = provider.provided_models(cx);
            assert_eq!(models.len(), 1);
            assert_eq!(
                models.first().map(|model| model.id().0.to_string()),
                Some("gpt-account-default".to_string())
            );
            assert_eq!(
                provider
                    .default_model(cx)
                    .map(|model| model.id().0.to_string()),
                Some("gpt-account-default".to_string())
            );
        });
    }

    struct FakeCredentialsProvider {
        storage: Mutex<Option<(String, Vec<u8>)>>,
    }

    impl FakeCredentialsProvider {
        fn new() -> Self {
            Self {
                storage: Mutex::new(None),
            }
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async { Ok(self.storage.lock().clone()) })
        }

        fn write_credentials<'a>(
            &'a self,
            _url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            self.storage
                .lock()
                .replace((username.to_string(), password.to_vec()));
            Box::pin(async { Ok(()) })
        }

        fn delete_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            *self.storage.lock() = None;
            Box::pin(async { Ok(()) })
        }
    }
}
