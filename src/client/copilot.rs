use super::openai::*;
use super::*;

use crate::config::{Config, CopilotAuthState};
use crate::utils::now_timestamp;

use anyhow::{anyhow, bail, Context, Result};
use reqwest::{Client as ReqwestClient, StatusCode};
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;

const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const COPILOT_API_BASE: &str = "https://api.githubcopilot.com";
const OAUTH_SCOPE: &str = "read:user";

const HEADER_EDITOR_VERSION: &str = "vscode/1.105.1";
const HEADER_EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.32.4";
const HEADER_USER_AGENT: &str = "GitHubCopilotChat/0.32.4";
const HEADER_COPILOT_INTEGRATION_ID: &str = "vscode-chat";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CopilotConfig {
    pub name: Option<String>,
    pub api_base: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl CopilotClient {
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 0] = [];

    fn api_base(&self) -> String {
        self.get_api_base()
            .unwrap_or_else(|_| COPILOT_API_BASE.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    async fn ensure_copilot_api_token(&self, client: &ReqwestClient) -> Result<String> {
        let mut state = Config::load_github_copilot_auth_state()?.unwrap_or_default();

        if let Some(token) = valid_cached_copilot_token(&state) {
            return Ok(token);
        }

        let github_oauth_token = match state.github_oauth_token.clone() {
            Some(v) if !v.trim().is_empty() => v,
            _ => {
                let token = start_device_flow(client).await?;
                state.github_oauth_token = Some(token.clone());
                Config::save_github_copilot_auth_state(&state)?;
                token
            }
        };

        let token_data = exchange_copilot_token(client, &github_oauth_token).await?;

        state.github_oauth_token = Some(github_oauth_token);
        state.copilot_api_token = Some(token_data.token.clone());
        state.expires_at = Some(token_data.expires_at);
        state.refresh_in = token_data.refresh_in;
        state.obtained_at = Some(now_timestamp());
        Config::save_github_copilot_auth_state(&state)?;

        Ok(token_data.token)
    }
}

#[async_trait::async_trait]
impl Client for CopilotClient {
    client_common_fns!();

    async fn chat_completions_inner(
        &self,
        client: &ReqwestClient,
        data: ChatCompletionsData,
    ) -> Result<ChatCompletionsOutput> {
        let token = self.ensure_copilot_api_token(client).await?;
        let request_data = prepare_chat_completions(self, data, &token)?;
        let builder = self.request_builder(client, request_data);
        openai_chat_completions(builder, self.model()).await
    }

    async fn chat_completions_streaming_inner(
        &self,
        client: &ReqwestClient,
        handler: &mut SseHandler,
        data: ChatCompletionsData,
    ) -> Result<()> {
        let token = self.ensure_copilot_api_token(client).await?;
        let request_data = prepare_chat_completions(self, data, &token)?;
        let builder = self.request_builder(client, request_data);
        openai_chat_completions_streaming(builder, handler, self.model()).await
    }

    async fn embeddings_inner(
        &self,
        client: &ReqwestClient,
        data: &EmbeddingsData,
    ) -> Result<EmbeddingsOutput> {
        let token = self.ensure_copilot_api_token(client).await?;
        let request_data = prepare_embeddings(self, data, &token)?;
        let builder = self.request_builder(client, request_data);
        openai_embeddings(builder, self.model()).await
    }

    async fn rerank_inner(
        &self,
        _client: &ReqwestClient,
        _data: &RerankData,
    ) -> Result<RerankOutput> {
        bail!("The client doesn't support rerank api")
    }
}

fn prepare_chat_completions(
    self_: &CopilotClient,
    data: ChatCompletionsData,
    token: &str,
) -> Result<RequestData> {
    let body = openai_build_chat_completions_body(data, &self_.model);
    let url = format!("{}/chat/completions", self_.api_base());
    let mut request_data = RequestData::new(url, body);
    request_data.bearer_auth(token);
    apply_copilot_identity_headers(&mut request_data);
    Ok(request_data)
}

fn prepare_embeddings(self_: &CopilotClient, data: &EmbeddingsData, token: &str) -> Result<RequestData> {
    let body = openai_build_embeddings_body(data, &self_.model);
    let url = format!("{}/embeddings", self_.api_base());
    let mut request_data = RequestData::new(url, body);
    request_data.bearer_auth(token);
    apply_copilot_identity_headers(&mut request_data);
    Ok(request_data)
}

fn apply_copilot_identity_headers(request_data: &mut RequestData) {
    request_data.header("Editor-Version", HEADER_EDITOR_VERSION);
    request_data.header("Editor-Plugin-Version", HEADER_EDITOR_PLUGIN_VERSION);
    request_data.header("User-Agent", HEADER_USER_AGENT);
    request_data.header("Copilot-Integration-Id", HEADER_COPILOT_INTEGRATION_ID);
}

fn valid_cached_copilot_token(state: &CopilotAuthState) -> Option<String> {
    let token = state.copilot_api_token.as_ref()?.trim();
    let expires_at = state.expires_at?;
    let now = now_timestamp();

    let obtained_at = state
        .obtained_at
        .or_else(|| state.refresh_in.map(|v| expires_at - v))
        .unwrap_or(expires_at - 1800);

    let lifetime = (expires_at - obtained_at).max(1);
    let refresh_at = expires_at - (lifetime / 10);
    if now < refresh_at {
        Some(token.to_string())
    } else {
        None
    }
}

async fn start_device_flow(client: &ReqwestClient) -> Result<String> {
    let device = request_device_code(client).await?;

    eprintln!(
        "GitHub Copilot login required. Open {} and enter code: {}",
        device.verification_uri, device.user_code
    );

    let mut interval = device.interval.max(1);
    let deadline = now_timestamp() + i64::from(device.expires_in.max(1));

    loop {
        if now_timestamp() >= deadline {
            bail!("Device login timed out before authorization completed")
        }

        tokio::time::sleep(Duration::from_secs(u64::from(interval))).await;

        let token = poll_access_token(client, &device.device_code).await?;
        match token {
            PollAccessTokenResult::Authorized(token) => return Ok(token),
            PollAccessTokenResult::AuthorizationPending => continue,
            PollAccessTokenResult::SlowDown => {
                interval += 5;
                continue;
            }
            PollAccessTokenResult::ExpiredToken => {
                bail!("Device login code expired. Please retry the login flow.")
            }
            PollAccessTokenResult::AccessDenied => {
                bail!("GitHub authorization denied by user")
            }
            PollAccessTokenResult::UnknownError(error) => {
                bail!("GitHub device authorization failed: {error}")
            }
        }
    }
}

async fn request_device_code(client: &ReqwestClient) -> Result<DeviceCodeResponse> {
    let response = client
        .post(DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .form(&[
            ("client_id", CLIENT_ID),
            ("scope", OAUTH_SCOPE),
        ])
        .send()
        .await
        .with_context(|| "Failed to initiate GitHub device authorization")?;

    let status = response.status();
    let data: Value = response
        .json()
        .await
        .with_context(|| "Failed to decode GitHub device authorization response")?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let output: DeviceCodeResponse =
        serde_json::from_value(data).with_context(|| "Invalid GitHub device authorization payload")?;
    Ok(output)
}

async fn poll_access_token(client: &ReqwestClient, device_code: &str) -> Result<PollAccessTokenResult> {
    let response = client
        .post(ACCESS_TOKEN_URL)
        .header("Accept", "application/json")
        .form(&[
            ("client_id", CLIENT_ID),
            ("device_code", device_code),
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:device_code",
            ),
        ])
        .send()
        .await
        .with_context(|| "Failed while polling GitHub access token")?;

    let status = response.status();
    let data: PollAccessTokenResponse = response
        .json()
        .await
        .with_context(|| "Failed to decode GitHub access token response")?;

    if status.is_success() {
        if let Some(token) = data.access_token.filter(|v| !v.trim().is_empty()) {
            return Ok(PollAccessTokenResult::Authorized(token));
        }
        if let Some(error) = data.error {
            return Ok(map_poll_error(&error));
        }
        return Err(anyhow!("GitHub access token response missing access_token"));
    }

    if let Some(error) = data.error {
        Ok(map_poll_error(&error))
    } else {
        Err(anyhow!("GitHub access token polling failed with status {status}"))
    }
}

fn map_poll_error(error: &str) -> PollAccessTokenResult {
    match error {
        "authorization_pending" => PollAccessTokenResult::AuthorizationPending,
        "slow_down" => PollAccessTokenResult::SlowDown,
        "expired_token" => PollAccessTokenResult::ExpiredToken,
        "access_denied" => PollAccessTokenResult::AccessDenied,
        _ => PollAccessTokenResult::UnknownError(error.to_string()),
    }
}

async fn exchange_copilot_token(client: &ReqwestClient, github_oauth_token: &str) -> Result<CopilotTokenResponse> {
    let response = client
        .get(COPILOT_TOKEN_URL)
        .header("Authorization", format!("Bearer {github_oauth_token}"))
        .header("Accept", "application/json")
        .header("Editor-Version", HEADER_EDITOR_VERSION)
        .header("Editor-Plugin-Version", HEADER_EDITOR_PLUGIN_VERSION)
        .header("User-Agent", HEADER_USER_AGENT)
        .header("Copilot-Integration-Id", HEADER_COPILOT_INTEGRATION_ID)
        .send()
        .await
        .with_context(|| "Failed to exchange GitHub token for Copilot API token")?;

    let status = response.status();
    let data: Value = response
        .json()
        .await
        .with_context(|| "Failed to decode Copilot token exchange response")?;

    if status == StatusCode::FORBIDDEN {
        bail!("GitHub Copilot is not available for this account (403)")
    }
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }

    let output: CopilotTokenResponse =
        serde_json::from_value(data).with_context(|| "Invalid Copilot token exchange payload")?;
    if output.token.trim().is_empty() {
        bail!("Copilot token exchange returned an empty token")
    }
    Ok(output)
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: u32,
    interval: u32,
}

#[derive(Debug, Deserialize)]
struct PollAccessTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
}

enum PollAccessTokenResult {
    Authorized(String),
    AuthorizationPending,
    SlowDown,
    ExpiredToken,
    AccessDenied,
    UnknownError(String),
}

#[derive(Debug, Deserialize)]
struct CopilotTokenResponse {
    token: String,
    expires_at: i64,
    refresh_in: Option<i64>,
}
