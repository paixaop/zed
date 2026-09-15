use gpui::{App, Context, Task, Window};
use language_model::{
    LanguageModel, LanguageModelAccountUsage, LanguageModelUsageBucket, LanguageModelUsageWindow,
};
use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};
use ui::{ButtonLike, ContextMenu, PopoverMenu, Tooltip, prelude::*};
use util::ResultExt as _;

pub(crate) struct AccountUsageView {
    model: Arc<dyn LanguageModel>,
    usage: Option<LanguageModelAccountUsage>,
    error: Option<String>,
    loading: bool,
    refresh_due: bool,
    refresh_task: Option<Task<()>>,
}

impl AccountUsageView {
    pub(crate) fn new(model: Arc<dyn LanguageModel>, cx: &mut Context<Self>) -> Self {
        let mut view = Self {
            model,
            usage: None,
            error: None,
            loading: false,
            refresh_due: false,
            refresh_task: None,
        };
        view.refresh(false, cx);
        view
    }

    fn refresh(&mut self, force: bool, cx: &mut Context<Self>) {
        self.loading = true;
        self.refresh_due = false;
        cx.notify();
        self.refresh_task = Some(cx.spawn(async move |this, cx| {
            let Ok(Some(task)) = this.update(cx, |view, cx| view.model.account_usage(force, cx))
            else {
                return;
            };
            let result = task.await;
            if this
                .update(cx, |view, cx| {
                    view.loading = false;
                    match result {
                        Ok(usage) => {
                            view.usage = Some(usage);
                            view.error = None;
                        }
                        Err(error) => {
                            log::warn!("Could not refresh account usage: {error:#}");
                            view.error = Some(usage_error_label(&error.to_string()).into());
                        }
                    }
                    cx.notify();
                })
                .is_err()
            {
                return;
            }
            cx.background_executor()
                .timer(Duration::from_secs(60))
                .await;
            // Hidden panes stop polling until they render again.
            this.update(cx, |view, cx| {
                view.refresh_due = true;
                cx.notify();
            })
            .log_err();
        }));
    }
}

fn bucket_status(bucket: &LanguageModelUsageBucket) -> Option<&'static str> {
    if bucket.limit_reached == Some(true) {
        Some(if bucket.name == "Included usage" {
            "Included quota exhausted"
        } else {
            "Quota exhausted"
        })
    } else if bucket.allowed == Some(false) {
        Some(if bucket.name == "Included usage" {
            "Included usage unavailable"
        } else {
            "Usage currently unavailable"
        })
    } else if bucket.allowed.is_none() {
        Some("Availability unknown")
    } else {
        None
    }
}

fn remaining_label(window: &LanguageModelUsageWindow) -> String {
    match window.remaining_percent {
        Some(percent) => format!("{}: {percent:.0}% left", window.label),
        None => format!("{}: unavailable", window.label),
    }
}

fn usage_error_label(error: &str) -> &'static str {
    let error = error.to_ascii_lowercase();
    if error.contains("connect") || error.contains("dns") {
        "connection error"
    } else if error.contains("timed out") || error.contains("timeout") {
        "request timed out"
    } else if error.contains("401")
        || error.contains("authentication")
        || error.contains("reauth")
        || error.contains("invalid_grant")
    {
        "sign-in required"
    } else if error.contains("403") {
        "access denied"
    } else if error.contains("429") {
        "too many requests"
    } else if error.contains("invalid chatgpt usage response")
        || error.contains("different account")
    {
        "invalid response"
    } else {
        "could not refresh"
    }
}

fn reset_label(window: &LanguageModelUsageWindow) -> Option<String> {
    reset_label_at(window, SystemTime::now())
}

fn reset_label_at(window: &LanguageModelUsageWindow, now: SystemTime) -> Option<String> {
    let reset = window.resets_at?;
    let Ok(duration) = reset.duration_since(now) else {
        return Some("Reset time passed; refresh to update".into());
    };
    let minutes = duration.as_secs().div_ceil(60).max(1);
    let countdown = if minutes >= 1440 {
        format!("{}d {}h", minutes / 1440, (minutes % 1440) / 60)
    } else if minutes >= 60 {
        format!("{}h {}m", minutes / 60, minutes % 60)
    } else {
        format!("{minutes}m")
    };
    let seconds =
        i64::try_from(reset.duration_since(SystemTime::UNIX_EPOCH).ok()?.as_secs()).ok()?;
    let date = chrono::DateTime::from_timestamp(seconds, 0)?.with_timezone(&chrono::Local);
    Some(format!(
        "Resets on {} ({countdown})",
        date.format("%-m/%-d/%Y at %-I:%M%p")
    ))
}

fn meter(
    window: &LanguageModelUsageWindow,
    muted: bool,
    compact: bool,
    cx: &App,
) -> impl IntoElement {
    let color = if muted {
        cx.theme().colors().text_muted
    } else if window
        .remaining_percent
        .is_some_and(|percent| percent <= 10.0)
    {
        cx.theme().status().warning
    } else {
        cx.theme().status().info
    };
    v_flex()
        .gap_0p5()
        .child(
            Label::new(if compact {
                window
                    .remaining_percent
                    .map(|percent| format!("{percent:.0}%"))
                    .unwrap_or_else(|| "—".into())
            } else {
                remaining_label(window)
            })
            .size(LabelSize::XSmall)
            .color(if muted { Color::Muted } else { Color::Default }),
        )
        .when_some(window.remaining_percent, |this, percent| {
            this.child(
                div()
                    .h(px(3.))
                    .w_full()
                    .rounded_full()
                    .bg(cx.theme().colors().element_background)
                    .child(
                        div()
                            .h_full()
                            .w(relative((percent / 100.0) as f32))
                            .rounded_full()
                            .bg(color),
                    ),
            )
        })
}

impl Render for AccountUsageView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.refresh_due {
            self.refresh(false, cx);
        }
        let usage = self.usage.clone();
        let error = self.error.clone();
        let stale = usage.as_ref().is_some_and(|usage| {
            error.is_some()
                || usage
                    .fetched_at
                    .elapsed()
                    .is_ok_and(|elapsed| elapsed > Duration::from_secs(120))
        });
        let included = usage.as_ref().and_then(|usage| usage.buckets.first());
        let status = included.and_then(bucket_status);
        let notice = usage
            .as_ref()
            .and_then(|usage| usage.warnings.first().cloned())
            .or_else(|| status.map(str::to_owned))
            .or_else(|| {
                usage
                    .as_ref()?
                    .buckets
                    .iter()
                    .skip(1)
                    .find(|bucket| {
                        bucket.limit_reached == Some(true) || bucket.allowed == Some(false)
                    })
                    .map(|bucket| format!("{}: unavailable", bucket.name))
            });
        let warning = error.is_some() || notice.is_some();
        let weak_view = cx.entity().downgrade();
        let mut tooltip = vec!["Selected account usage".to_owned()];
        if let Some(included) = included {
            tooltip.extend(included.windows.iter().map(remaining_label));
        }
        if self.loading {
            tooltip.push("Refreshing usage…".into());
        }
        if stale {
            tooltip.push("Stale usage; refresh to update".into());
        }
        if let Some(notice) = notice {
            tooltip.push(notice);
        }
        let trigger = ButtonLike::new("account-usage-trigger").child(
            h_flex()
                .gap_2()
                .when_some(included, |this, bucket| {
                    this.children(
                        bucket
                            .windows
                            .iter()
                            .map(|window| meter(window, stale || warning, true, cx)),
                    )
                })
                .when(
                    included.is_none_or(|bucket| bucket.windows.is_empty()),
                    |this| this.child(Label::new("—").size(LabelSize::XSmall).color(Color::Muted)),
                )
                .when(stale, |this| {
                    this.child(
                        Icon::new(IconName::HistoryRerun)
                            .size(IconSize::XSmall)
                            .color(Color::Warning),
                    )
                })
                .when(warning, |this| {
                    this.child(
                        Icon::new(IconName::Warning)
                            .size(IconSize::XSmall)
                            .color(Color::Warning),
                    )
                }),
        );
        PopoverMenu::new("account-usage")
            .trigger_with_tooltip(trigger, Tooltip::text(tooltip.join("\n")))
            .menu(move |window, cx| {
                let usage = usage.clone();
                let error = error.clone();
                let weak_view = weak_view.clone();
                Some(ContextMenu::build(window, cx, move |mut menu, _, _| {
                    menu = menu.header("ChatGPT account usage");
                    if let Some(error) = error { menu = menu.label(format!("Usage unavailable: {error}")); }
                    if let Some(usage) = usage {
                        if let Some(plan) = usage.plan { menu = menu.label(format!("Plan: {plan}")); }
                        if let Ok(age) = usage.fetched_at.elapsed() {
                            menu = menu.label(format!("Updated {} seconds ago", age.as_secs()));
                        }
                        for warning in usage.warnings { menu = menu.label(warning); }
                        for bucket in usage.buckets {
                            menu = menu.separator().header(bucket.name.clone());
                            if let Some(status) = bucket_status(&bucket) { menu = menu.label(status); }
                            if bucket.windows.is_empty() { menu = menu.label("Window data unavailable"); }
                            for window in bucket.windows {
                                let reset = reset_label(&window);
                                menu = menu.custom_entry(move |_, cx| meter(&window, stale, false, cx).into_any_element(), |_, _| {}).selectable(false);
                                if let Some(reset) = reset { menu = menu.label(reset); }
                            }
                        }
                        menu = menu.separator();
                        for detail in usage.details { menu = menu.label(detail); }
                        menu = menu.label("Included quota and purchased credits are separate.")
                            .label("Usage may lag recent requests. Extra buckets have their own limits.");
                    }
                    menu.separator().entry("Refresh usage", None, move |_, cx| {
                        weak_view.update(cx, |view, cx| view.refresh(true, cx)).log_err();
                    })
                }))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_flags_override_remaining_percentages() {
        let mut bucket = LanguageModelUsageBucket {
            name: "Included usage".into(),
            allowed: Some(true),
            limit_reached: Some(false),
            windows: vec![LanguageModelUsageWindow {
                label: "5h".into(),
                remaining_percent: Some(75.),
                resets_at: None,
            }],
        };
        assert_eq!(bucket_status(&bucket), None);
        bucket.allowed = None;
        assert_eq!(bucket_status(&bucket), Some("Availability unknown"));
        bucket.allowed = Some(false);
        assert_eq!(bucket_status(&bucket), Some("Included usage unavailable"));
        bucket.limit_reached = Some(true);
        assert_eq!(bucket_status(&bucket), Some("Included quota exhausted"));
        assert_eq!(
            remaining_label(bucket.windows.first().expect("window")),
            "5h: 75% left"
        );
    }

    #[test]
    fn reset_date_uses_local_time_and_keeps_countdown() {
        use chrono::TimeZone as _;
        let reset = chrono::Local
            .with_ymd_and_hms(2026, 9, 19, 1, 0, 0)
            .single()
            .expect("local date");
        let reset = SystemTime::from(reset);
        let window = LanguageModelUsageWindow {
            label: "Weekly".into(),
            remaining_percent: Some(0.),
            resets_at: Some(reset),
        };
        assert_eq!(
            reset_label_at(&window, reset - Duration::from_secs(5 * 86400 + 3600)).as_deref(),
            Some("Resets on 9/19/2026 at 1:00AM (5d 1h)")
        );
    }

    #[test]
    fn usage_errors_hide_raw_request_details() {
        for (error, expected) in [
            (
                "error sending request for url (https://chatgpt.com/backend-api/wham/usage): client error (Connect): operation timed out",
                "connection error",
            ),
            ("ChatGPT usage request timed out", "request timed out"),
            (
                "Could not load ChatGPT account usage (HTTP 401 Unauthorized)",
                "sign-in required",
            ),
            (
                "Could not load ChatGPT account usage (HTTP 429 Too Many Requests)",
                "too many requests",
            ),
            (
                "Invalid ChatGPT usage response: private details",
                "invalid response",
            ),
            ("unexpected private details", "could not refresh"),
        ] {
            assert_eq!(usage_error_label(error), expected);
        }
    }

    #[test]
    fn absent_percentage_and_passed_reset_do_not_show_full_quota() {
        let window = LanguageModelUsageWindow {
            label: "Weekly".into(),
            remaining_percent: None,
            resets_at: Some(SystemTime::UNIX_EPOCH),
        };
        assert_eq!(remaining_label(&window), "Weekly: unavailable");
        assert_eq!(
            reset_label(&window).as_deref(),
            Some("Reset time passed; refresh to update")
        );
    }
}
