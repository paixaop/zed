use super::*;
use language_model::{
    LanguageModelAccountUsage, LanguageModelUsageBucket, LanguageModelUsageWindow,
};

const USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";
const CACHE_DURATION: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

type UsageResult = Result<LanguageModelAccountUsage, Arc<anyhow::Error>>;
pub(super) struct UsageRequest {
    started_at: std::time::Instant,
    task: Shared<Task<UsageResult>>,
}

#[derive(Default, Deserialize)]
struct UsageResponse {
    account_id: Option<String>,
    #[serde(default, deserialize_with = "optional_display_text")]
    plan_type: Option<String>,
    rate_limit: Option<RateLimit>,
    code_review_rate_limit: Option<RateLimit>,
    additional_rate_limits: Option<Vec<AdditionalRateLimit>>,
    credits: Option<Credits>,
    spend_control: Option<SpendControl>,
    #[serde(default, deserialize_with = "optional_display_text")]
    rate_limit_reached_type: Option<String>,
    rate_limit_reset_credits: Option<ResetCredits>,
}

#[derive(Default, Deserialize)]
struct RateLimit {
    allowed: Option<bool>,
    limit_reached: Option<bool>,
    primary_window: Option<UsageWindow>,
    secondary_window: Option<UsageWindow>,
}

#[derive(Default, Deserialize)]
struct UsageWindow {
    used_percent: Option<f64>,
    limit_window_seconds: Option<u64>,
    reset_after_seconds: Option<u64>,
    reset_at: Option<u64>,
}

#[derive(Deserialize)]
struct AdditionalRateLimit {
    #[serde(default, deserialize_with = "optional_display_text")]
    limit_name: Option<String>,
    #[serde(default, deserialize_with = "optional_display_text")]
    metered_feature: Option<String>,
    rate_limit: Option<RateLimit>,
}

#[derive(Deserialize)]
struct Credits {
    has_credits: Option<bool>,
    unlimited: Option<bool>,
    overage_limit_reached: Option<bool>,
    #[serde(default, deserialize_with = "optional_display_text")]
    balance: Option<String>,
}

#[derive(Deserialize)]
struct SpendControl {
    reached: Option<bool>,
    individual_limit: Option<IndividualLimit>,
}

#[derive(Deserialize)]
struct IndividualLimit {
    #[serde(default, deserialize_with = "optional_display_text")]
    remaining: Option<String>,
    #[serde(default, deserialize_with = "optional_display_text")]
    limit: Option<String>,
    used_percent: Option<f64>,
    remaining_percent: Option<f64>,
    reset_after_seconds: Option<u64>,
    reset_at: Option<u64>,
}

#[derive(Deserialize)]
struct ResetCredits {
    available_count: Option<u64>,
}

// WHAM's optional display metadata can change shape independently of its quota windows.
// Preserve scalar values, but never let an opaque object hide the account's usage.
fn optional_display_text<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::String(value) => Some(value),
        serde_json::Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn reset_time(
    absolute: Option<u64>,
    relative: Option<u64>,
    fetched_at: SystemTime,
) -> Option<SystemTime> {
    // Relative time uses the server's clock and avoids depending on the local clock's accuracy.
    relative
        .and_then(|seconds| fetched_at.checked_add(Duration::from_secs(seconds)))
        .or_else(|| {
            absolute.and_then(|seconds| UNIX_EPOCH.checked_add(Duration::from_secs(seconds)))
        })
}

fn window_label(seconds: Option<u64>) -> String {
    match seconds {
        Some(604800) => "Weekly".into(),
        Some(seconds) if seconds > 0 && seconds.is_multiple_of(86400) => {
            format!("{}d", seconds / 86400)
        }
        Some(seconds) if seconds > 0 && seconds.is_multiple_of(3600) => {
            format!("{}h", seconds / 3600)
        }
        Some(seconds) if seconds > 0 && seconds.is_multiple_of(60) => format!("{}m", seconds / 60),
        Some(seconds) if seconds > 0 => format!("{seconds}s"),
        _ => "Window".into(),
    }
}

impl RateLimit {
    fn normalize(self, name: String, fetched_at: SystemTime) -> LanguageModelUsageBucket {
        LanguageModelUsageBucket {
            name,
            allowed: self.allowed,
            limit_reached: self.limit_reached,
            windows: [self.primary_window, self.secondary_window]
                .into_iter()
                .flatten()
                .map(|window| LanguageModelUsageWindow {
                    label: window_label(window.limit_window_seconds),
                    remaining_percent: window
                        .used_percent
                        .filter(|used| used.is_finite())
                        .map(|used| (100.0 - used).clamp(0.0, 100.0)),
                    resets_at: reset_time(window.reset_at, window.reset_after_seconds, fetched_at),
                })
                .collect(),
        }
    }
}

impl UsageResponse {
    fn normalize(self, fetched_at: SystemTime) -> LanguageModelAccountUsage {
        let mut usage = LanguageModelAccountUsage {
            plan: self.plan_type,
            buckets: vec![
                self.rate_limit
                    .unwrap_or_default()
                    .normalize("Included usage".into(), fetched_at),
            ],
            details: Vec::new(),
            warnings: Vec::new(),
            fetched_at,
        };
        if let Some(limit) = self.code_review_rate_limit {
            usage
                .buckets
                .push(limit.normalize("Code review".into(), fetched_at));
        }
        for additional in self.additional_rate_limits.unwrap_or_default() {
            let name = additional
                .limit_name
                .or(additional.metered_feature)
                .unwrap_or_else(|| "Additional usage".into());
            usage.buckets.push(
                additional
                    .rate_limit
                    .unwrap_or_default()
                    .normalize(name, fetched_at),
            );
        }
        if let Some(credits) = self.credits {
            usage.details.push(
                match (credits.unlimited, credits.has_credits) {
                    (Some(true), _) => "Purchased-credit availability: unlimited",
                    (_, Some(true)) => "Purchased-credit availability: available",
                    (_, Some(false)) => "Purchased-credit availability: unavailable",
                    (_, None) => "Purchased-credit availability: unknown",
                }
                .into(),
            );
            if credits.unlimited == Some(true) {
                usage.details.push("Purchased credits: unlimited".into());
            } else if let Some(balance) = credits.balance {
                usage.details.push(format!("Purchased credits: {balance}"));
            } else {
                usage.details.push(
                    match credits.has_credits {
                        Some(true) => "Purchased credits available; balance unknown",
                        Some(false) => "No purchased credits available",
                        None => "Purchased credits: unknown",
                    }
                    .into(),
                );
            }
            if credits.overage_limit_reached == Some(true) {
                usage.warnings.push("Purchased-credit cap reached".into());
            }
        }
        if let Some(control) = self.spend_control {
            if control.reached == Some(true) {
                usage.warnings.push("Monthly spend cap reached".into());
            }
            if let Some(limit) = control.individual_limit {
                if let Some(remaining) = limit.remaining {
                    usage
                        .details
                        .push(format!("Monthly credits remaining: {remaining}"));
                }
                if let Some(limit) = limit.limit {
                    usage
                        .details
                        .push(format!("Monthly credit allowance: {limit}"));
                }
                usage.buckets.push(LanguageModelUsageBucket {
                    name: "Monthly credit limit".into(),
                    allowed: control.reached.map(|reached| !reached),
                    limit_reached: control.reached,
                    windows: vec![LanguageModelUsageWindow {
                        label: "Monthly".into(),
                        remaining_percent: limit
                            .remaining_percent
                            .or_else(|| limit.used_percent.map(|used| 100.0 - used))
                            .filter(|remaining| remaining.is_finite())
                            .map(|remaining| remaining.clamp(0.0, 100.0)),
                        resets_at: reset_time(
                            limit.reset_at,
                            limit.reset_after_seconds,
                            fetched_at,
                        ),
                    }],
                });
            }
        }
        if let Some(reason) = self.rate_limit_reached_type {
            usage.details.push(format!("Limiter: {reason}"));
        }
        if let Some(count) = self
            .rate_limit_reset_credits
            .and_then(|credits| credits.available_count)
        {
            usage
                .details
                .push(format!("Available usage resets: {count}"));
        }
        usage
    }
}

pub(super) fn load(
    state: &mut State,
    id: String,
    force: bool,
    cx: &mut Context<State>,
) -> Task<Result<LanguageModelAccountUsage>> {
    if let Some(request) = state.usage_requests.get(&id)
        && (request.task.peek().is_none()
            || (!force && request.started_at.elapsed() < CACHE_DURATION))
    {
        let task = request.task.clone();
        return cx.spawn(async move |_, _| task.await.map_err(|error| anyhow!("{error:#}")));
    }
    let http_client = state.http_client.clone();
    let generation = state.auth_generation;
    let account_id = id.clone();
    let task = cx
        .spawn(async move |state, cx| {
            let result = fetch(&state, &http_client, &account_id, cx).await;
            if state.read_with(cx, |state, _| {
                state.auth_generation != generation
                    || !state.accounts.accounts.contains_key(&account_id)
            })? {
                return Err(Arc::new(anyhow!("Account changed while loading usage")));
            }
            result.map_err(Arc::new)
        })
        .shared();
    state.usage_requests.insert(
        id,
        UsageRequest {
            started_at: std::time::Instant::now(),
            task: task.clone(),
        },
    );
    cx.spawn(async move |_, _| task.await.map_err(|error| anyhow!("{error:#}")))
}

async fn fetch(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    id: &str,
    cx: &mut AsyncApp,
) -> Result<LanguageModelAccountUsage> {
    let mut credentials = account_credentials(state, http_client, id, None, cx).await?;
    let mut refreshed = false;
    let mut retried = false;
    loop {
        let result = async {
            let headers = codex_extra_headers(&credentials, None);
            let request = HttpRequest::builder()
                .method(Method::GET)
                .uri(USAGE_URL)
                .header("Accept", "application/json")
                .header(
                    "Authorization",
                    format!("Bearer {}", credentials.access_token),
                )
                .extra_headers(&headers)
                .body(AsyncBody::default())?;
            let mut response = http_client.send(request).await?;
            let status = response.status();
            let mut body = String::new();
            smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;
            anyhow::Ok((status, body))
        };
        let timeout = cx.background_executor().timer(REQUEST_TIMEOUT);
        let result = futures::select! {
            result = result.fuse() => result,
            () = timeout.fuse() => Err(anyhow!("ChatGPT usage request timed out")),
        };
        if !retried
            && result
                .as_ref()
                .map_or(true, |(status, _)| status.is_server_error())
        {
            retried = true;
            cx.background_executor().timer(Duration::from_secs(1)).await;
            continue;
        }
        let (status, body) = result?;
        if status == http_client::StatusCode::UNAUTHORIZED && !refreshed {
            refreshed = true;
            credentials =
                account_credentials(state, http_client, id, Some(&credentials.access_token), cx)
                    .await?;
            continue;
        }
        anyhow::ensure!(
            status.is_success(),
            "Could not load ChatGPT account usage (HTTP {status})"
        );
        let response: UsageResponse =
            serde_json::from_str(&body).context("Invalid ChatGPT usage response")?;
        if let Some(returned) = &response.account_id
            && let Some(expected) = &credentials.account_id
        {
            anyhow::ensure!(
                returned == expected,
                "Usage response belongs to a different account"
            );
        }
        return Ok(response.normalize(SystemTime::now()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AppContext as _, TestAppContext};
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::{future::Future, pin::Pin};

    #[test]
    fn normalizes_both_windows_and_server_clock() {
        let response: UsageResponse = serde_json::from_value(serde_json::json!({
            "plan_type": "pro", "rate_limit": { "allowed": true, "limit_reached": false,
                "primary_window": { "used_percent": 25, "limit_window_seconds": 18000, "reset_after_seconds": 7200, "reset_at": 123 },
                "secondary_window": { "used_percent": 9, "limit_window_seconds": 604800, "reset_after_seconds": 432000 }
            }, "promo": { "new_unknown_key": true }
        })).expect("payload");
        let fetched_at = UNIX_EPOCH + Duration::from_secs(1000);
        let usage = response.normalize(fetched_at);
        let included = usage.buckets.first().expect("included");
        assert_eq!(included.allowed, Some(true));
        assert_eq!(included.windows.len(), 2);
        let primary = included.windows.first().expect("primary");
        assert_eq!(primary.label, "5h");
        assert_eq!(primary.remaining_percent, Some(75.));
        assert_eq!(
            primary.resets_at,
            Some(fetched_at + Duration::from_secs(7200))
        );
        let secondary = included.windows.get(1).expect("secondary");
        assert_eq!(secondary.label, "Weekly");
        assert_eq!(secondary.remaining_percent, Some(91.));
    }

    #[test]
    fn free_and_missing_windows_are_not_unlimited() {
        for payload in [
            serde_json::json!({ "plan_type": "free", "rate_limit": { "primary_window": { "used_percent": 60, "limit_window_seconds": 604800 }, "secondary_window": null } }),
            serde_json::json!({ "rate_limit": null, "credits": null, "spend_control": null, "additional_rate_limits": null }),
        ] {
            let usage = serde_json::from_value::<UsageResponse>(payload)
                .expect("payload")
                .normalize(UNIX_EPOCH);
            let included = usage.buckets.first().expect("included");
            assert_eq!(included.allowed, None);
            if usage.plan.as_deref() == Some("free") {
                assert_eq!(included.windows.len(), 1);
                assert_eq!(included.windows.first().expect("weekly").label, "Weekly");
                assert_eq!(
                    included.windows.first().expect("weekly").remaining_percent,
                    Some(40.)
                );
            } else {
                assert!(included.windows.is_empty());
            }
        }
    }

    #[test]
    fn preserves_flags_credits_and_extra_buckets() {
        let usage = serde_json::from_value::<UsageResponse>(serde_json::json!({
            "rate_limit": { "allowed": false, "limit_reached": true,
                "primary_window": { "used_percent": 25, "limit_window_seconds": 18000 },
                "secondary_window": { "used_percent": 100, "limit_window_seconds": 604800 }
            },
            "credits": { "has_credits": true, "balance": "766.76", "overage_limit_reached": true },
            "additional_rate_limits": [{ "limit_name": "Premium", "metered_feature": "premium", "rate_limit": { "allowed": false, "limit_reached": true, "primary_window": null, "secondary_window": null } }],
            "spend_control": { "reached": true, "individual_limit": { "remaining": "17000", "limit": "25000", "remaining_percent": 68, "reset_after_seconds": 86400 } },
            "rate_limit_reset_credits": { "available_count": 2 }
        })).expect("payload").normalize(UNIX_EPOCH);
        let included = usage.buckets.first().expect("included");
        assert_eq!(included.allowed, Some(false));
        assert_eq!(included.limit_reached, Some(true));
        assert_eq!(
            included.windows.first().expect("primary").remaining_percent,
            Some(75.)
        );
        assert_eq!(
            included.windows.get(1).expect("weekly").remaining_percent,
            Some(0.)
        );
        let premium = usage.buckets.get(1).expect("premium");
        assert_eq!(premium.name, "Premium");
        assert_eq!(premium.limit_reached, Some(true));
        assert!(premium.windows.is_empty());
        assert!(
            usage
                .warnings
                .iter()
                .any(|warning| warning.contains("Monthly spend cap"))
        );
        assert!(
            usage
                .warnings
                .iter()
                .any(|warning| warning.contains("Purchased-credit cap"))
        );
        assert!(usage.details.iter().any(|detail| detail.contains("766.76")));
        assert_eq!(
            usage
                .buckets
                .get(2)
                .expect("monthly")
                .windows
                .first()
                .expect("limit")
                .remaining_percent,
            Some(68.)
        );
    }

    #[test]
    fn evolving_display_metadata_does_not_hide_usage() {
        for metadata in [
            serde_json::json!({"type": "weekly", "details": {"limit": "reached"}}),
            serde_json::json!(["weekly"]),
            serde_json::Value::Null,
        ] {
            let usage = serde_json::from_value::<UsageResponse>(serde_json::json!({
                "plan_type": metadata,
                "rate_limit_reached_type": metadata,
                "rate_limit": {
                    "allowed": false, "limit_reached": true,
                    "primary_window": {"used_percent": 25, "limit_window_seconds": 18000}
                },
                "additional_rate_limits": [{
                    "limit_name": metadata, "metered_feature": metadata,
                    "rate_limit": {"allowed": false, "limit_reached": true}
                }],
                "credits": {"balance": metadata},
                "spend_control": {"reached": true, "individual_limit": {
                    "remaining": metadata, "limit": metadata
                }}
            }))
            .expect("evolving metadata")
            .normalize(UNIX_EPOCH);
            let included = usage.buckets.first().expect("included");
            assert_eq!(included.allowed, Some(false));
            assert_eq!(included.limit_reached, Some(true));
            assert_eq!(
                included.windows.first().expect("window").remaining_percent,
                Some(75.)
            );
            assert_eq!(
                usage.buckets.get(1).expect("additional").name,
                "Additional usage"
            );
            assert!(
                usage
                    .warnings
                    .iter()
                    .any(|warning| warning == "Monthly spend cap reached")
            );
        }
    }

    #[test]
    fn display_metadata_accepts_strings_and_numeric_credit_amounts() {
        let usage = serde_json::from_value::<UsageResponse>(serde_json::json!({
            "plan_type": "pro",
            "rate_limit_reached_type": "weekly",
            "credits": {"balance": 766.76},
            "spend_control": {"individual_limit": {"remaining": 17000, "limit": "25000"}}
        }))
        .expect("scalar metadata")
        .normalize(UNIX_EPOCH);
        assert_eq!(usage.plan.as_deref(), Some("pro"));
        for detail in [
            "Limiter: weekly",
            "Purchased credits: 766.76",
            "Monthly credits remaining: 17000",
            "Monthly credit allowance: 25000",
        ] {
            assert!(usage.details.iter().any(|value| value == detail));
        }
        assert!(
            serde_json::from_value::<UsageResponse>(serde_json::json!({
                "account_id": {"unexpected": "object"}
            }))
            .is_err()
        );
    }

    #[test]
    fn clamps_percentages_and_preserves_unknown_values() {
        for (used, remaining) in [
            (Some(-20.), Some(100.)),
            (Some(120.), Some(0.)),
            (None, None),
        ] {
            let bucket = RateLimit {
                primary_window: Some(UsageWindow {
                    used_percent: used,
                    ..Default::default()
                }),
                ..Default::default()
            }
            .normalize("Included".into(), UNIX_EPOCH);
            assert_eq!(
                bucket.windows.first().expect("window").remaining_percent,
                remaining
            );
        }
    }

    #[gpui::test]
    async fn refreshes_selected_account_once_and_coalesces_requests(cx: &mut TestAppContext) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let http = FakeHttpClient::create({
            let calls = calls.clone();
            move |request| {
                let calls = calls.clone();
                async move {
                    let path = request.uri().path().to_owned();
                    let authorization = request
                        .headers()
                        .get("authorization")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_owned();
                    let account = request
                        .headers()
                        .get("chatgpt-account-id")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_owned();
                    calls
                        .lock()
                        .push((path.clone(), authorization.clone(), account));
                    let (status, body) = if path == "/oauth/token" {
                        (200, serde_json::json!({"access_token":"fresh", "refresh_token":"fresh_refresh", "expires_in":3600}).to_string())
                    } else if authorization == "Bearer first_token" {
                        (401, "invalid_token".into())
                    } else {
                        (200, serde_json::json!({ "account_id": "first", "rate_limit": { "allowed": true, "primary_window": { "used_percent": 12, "limit_window_seconds": 18000 } } }).to_string())
                    };
                    Ok(http_client::Response::builder()
                        .status(status)
                        .body(AsyncBody::from(body))?)
                }
            }
        });
        let state = make_state(http, cx);
        let model = cx
            .read(|cx| create_language_model(ChatGptModel::Gpt54, &state, cx))
            .with_account("first".into())
            .expect("bound model");
        let first = cx.update(|cx| model.account_usage(false, cx).expect("usage"));
        let concurrent = cx.update(|cx| model.account_usage(true, cx).expect("usage"));
        first.await.expect("first result");
        concurrent.await.expect("coalesced result");
        cx.update(|cx| model.account_usage(false, cx).expect("cached usage"))
            .await
            .expect("cached result");
        assert_eq!(
            *calls.lock(),
            vec![
                (
                    "/backend-api/wham/usage".into(),
                    "Bearer first_token".into(),
                    "first".into()
                ),
                ("/oauth/token".into(), "".into(), "".into()),
                (
                    "/backend-api/wham/usage".into(),
                    "Bearer fresh".into(),
                    "first".into()
                ),
            ]
        );
        cx.update(|cx| model.account_usage(true, cx).expect("refresh"))
            .await
            .expect("refresh result");
        assert_eq!(calls.lock().len(), 4);
    }

    #[gpui::test]
    async fn retries_transient_usage_failures_once_on_the_same_account(cx: &mut TestAppContext) {
        for failure in ["connection", "server", "persistent"] {
            let calls = Arc::new(Mutex::new(0));
            let http = FakeHttpClient::create({
                let calls = calls.clone();
                move |request| {
                    let calls = calls.clone();
                    async move {
                        assert_eq!(request.uri().path(), "/backend-api/wham/usage");
                        assert_eq!(
                            request
                                .headers()
                                .get("chatgpt-account-id")
                                .and_then(|value| value.to_str().ok()),
                            Some("first")
                        );
                        assert_eq!(
                            request
                                .headers()
                                .get("authorization")
                                .and_then(|value| value.to_str().ok()),
                            Some("Bearer first_token")
                        );
                        let count = {
                            let mut count = calls.lock();
                            *count += 1;
                            *count
                        };
                        if failure == "persistent" || (count == 1 && failure == "connection") {
                            return Err(anyhow!("connection timed out"));
                        }
                        Ok(http_client::Response::builder()
                            .status(if count == 1 { 503 } else { 200 })
                            .body(AsyncBody::from(
                                r#"{"account_id":"first","rate_limit":{"allowed":true}}"#,
                            ))?)
                    }
                }
            });
            let state = make_state(http, cx);
            let result = state
                .update(cx, |state, cx| load(state, "first".into(), false, cx))
                .await;
            assert_eq!(result.is_ok(), failure != "persistent");
            assert_eq!(*calls.lock(), 2);
        }
    }

    #[gpui::test]
    async fn usage_errors_do_not_fail_over_or_disable_completion_scopes(cx: &mut TestAppContext) {
        for status in [401, 403, 429] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let http = FakeHttpClient::create({
                let calls = calls.clone();
                move |request| {
                    let calls = calls.clone();
                    async move {
                        let path = request.uri().path().to_owned();
                        calls.lock().push(path.clone());
                        assert_ne!(
                            request
                                .headers()
                                .get("chatgpt-account-id")
                                .and_then(|value| value.to_str().ok()),
                            Some("second")
                        );
                        Ok(http_client::Response::builder()
                            .status(if path == "/oauth/token" { 400 } else { status })
                            .body(AsyncBody::from("invalid_grant"))?)
                    }
                }
            });
            let state = make_state(http, cx);
            assert!(
                state
                    .update(cx, |state, cx| load(state, "first".into(), false, cx))
                    .await
                    .is_err()
            );
            assert_eq!(calls.lock().len(), if status == 401 { 2 } else { 1 });
            cx.read(|cx| {
                assert!(
                    state
                        .read(cx)
                        .accounts
                        .accounts
                        .get("first")
                        .expect("first")
                        .blocked_models
                        .is_empty()
                )
            });
        }
    }

    #[gpui::test]
    async fn rejects_mismatched_account_and_times_out(cx: &mut TestAppContext) {
        let http = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(AsyncBody::from(r#"{"account_id":"second"}"#))?)
        });
        let state = make_state(http, cx);
        assert!(
            state
                .update(cx, |state, cx| load(state, "first".into(), false, cx))
                .await
                .expect_err("wrong account")
                .to_string()
                .contains("different account")
        );
        let http = FakeHttpClient::create(|_| async { futures::future::pending().await });
        let state = make_state(http, cx);
        assert!(
            state
                .update(cx, |state, cx| load(state, "first".into(), false, cx))
                .await
                .expect_err("timeout")
                .to_string()
                .contains("timed out")
        );
    }

    #[gpui::test]
    async fn sign_out_discards_pending_usage(cx: &mut TestAppContext) {
        let (sender, receiver) = futures::channel::oneshot::channel::<()>();
        let receiver = Arc::new(Mutex::new(Some(receiver)));
        let http = FakeHttpClient::create(move |_| {
            let receiver = receiver.lock().take().expect("one request");
            async move {
                receiver.await?;
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(AsyncBody::from(r#"{"account_id":"first"}"#))?)
            }
        });
        let state = make_state(http, cx);
        let usage = state.update(cx, |state, cx| load(state, "first".into(), false, cx));
        cx.run_until_parked();
        state
            .update(cx, |state, cx| state.sign_out(cx))
            .await
            .expect("sign out");
        sender.send(()).expect("release response");
        assert!(
            usage
                .await
                .expect_err("discard response")
                .to_string()
                .contains("Account changed")
        );
    }

    fn make_state(http: Arc<dyn HttpClient>, cx: &mut TestAppContext) -> Entity<State> {
        let state = cx.new(|cx| State::new(http, Arc::new(MemoryCredentials::default()), cx));
        state.update(cx, |state, _| {
            state.auth_generation = 1;
            for id in ["first", "second"] {
                state.accounts.insert(CodexCredentials {
                    access_token: format!("{id}_token"),
                    refresh_token: format!("{id}_refresh"),
                    account_id: Some(id.into()),
                    email: None,
                    expires_at_ms: u64::MAX,
                    scopes: Vec::new(),
                });
            }
        });
        state
    }

    #[derive(Default)]
    struct MemoryCredentials(Mutex<BTreeMap<String, (String, Vec<u8>)>>);
    impl CredentialsProvider for MemoryCredentials {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async move { Ok(self.0.lock().get(url).cloned()) })
        }
        fn write_credentials<'a>(
            &'a self,
            url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            Box::pin(async move {
                self.0
                    .lock()
                    .insert(url.into(), (username.into(), password.into()));
                Ok(())
            })
        }
        fn delete_credentials<'a>(
            &'a self,
            url: &'a str,
            _: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            Box::pin(async move {
                self.0.lock().remove(url);
                Ok(())
            })
        }
    }
}
