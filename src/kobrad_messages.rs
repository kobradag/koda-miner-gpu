use crate::proto::{
    kobrad_message::Payload, GetBlockTemplateRequestMessage, GetInfoRequestMessage, KobradMessage,
    NotifyBlockAddedRequestMessage, NotifyNewBlockTemplateRequestMessage, RpcBlock, SubmitBlockRequestMessage,
};
use crate::{
    pow::{self, HeaderHasher},
    Hash,
};

impl KobradMessage {
    #[inline(always)]
    pub fn get_info_request() -> Self {
        KobradMessage { payload: Some(Payload::GetInfoRequest(GetInfoRequestMessage {})) }
    }
    #[inline(always)]
    pub fn notify_block_added() -> Self {
        KobradMessage { payload: Some(Payload::NotifyBlockAddedRequest(NotifyBlockAddedRequestMessage {})) }
    }

    #[inline(always)]
    pub fn submit_block(block: RpcBlock) -> Self {
        KobradMessage {
            payload: Some(Payload::SubmitBlockRequest(SubmitBlockRequestMessage {
                block: Some(block),
                allow_non_daa_blocks: false,
            })),
        }
    }
}

impl From<GetInfoRequestMessage> for KobradMessage {
    fn from(a: GetInfoRequestMessage) -> Self {
        KobradMessage { payload: Some(Payload::GetInfoRequest(a)) }
    }
}
impl From<NotifyBlockAddedRequestMessage> for KobradMessage {
    fn from(a: NotifyBlockAddedRequestMessage) -> Self {
        KobradMessage { payload: Some(Payload::NotifyBlockAddedRequest(a)) }
    }
}

impl From<GetBlockTemplateRequestMessage> for KobradMessage {
    fn from(a: GetBlockTemplateRequestMessage) -> Self {
        KobradMessage { payload: Some(Payload::GetBlockTemplateRequest(a)) }
    }
}

impl From<NotifyNewBlockTemplateRequestMessage> for KobradMessage {
    fn from(a: NotifyNewBlockTemplateRequestMessage) -> Self {
        KobradMessage { payload: Some(Payload::NotifyNewBlockTemplateRequest(a)) }
    }
}

impl RpcBlock {
    #[inline(always)]
    pub fn block_hash(&self) -> Option<Hash> {
        let mut hasher = HeaderHasher::new();
        pow::serialize_header(&mut hasher, self.header.as_ref()?, false);
        Some(hasher.finalize())
    }
}
