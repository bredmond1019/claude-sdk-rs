use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct TaskContext {
    pub data: Arc<RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>>,
    pub metadata: Arc<RwLock<HashMap<String, Value>>>,
}

impl TaskContext {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn set<T: Any + Send + Sync + 'static>(&self, key: &str, value: T) {
        let mut data = self.data.write().await;
        data.insert(key.to_string(), Arc::new(value));
    }

    pub async fn get<T: Any + Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        let data = self.data.read().await;
        data.get(key)?.clone().downcast::<T>().ok()
    }

    pub async fn set_metadata(&self, key: &str, value: Value) {
        let mut metadata = self.metadata.write().await;
        metadata.insert(key.to_string(), value);
    }

    pub async fn get_metadata(&self, key: &str) -> Option<Value> {
        let metadata = self.metadata.read().await;
        metadata.get(key).cloned()
    }
}

impl Default for TaskContext {
    fn default() -> Self {
        Self::new()
    }
}
