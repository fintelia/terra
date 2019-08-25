use crate::terrain::quadtree::Node;
use serde::{Serialize, Deserialize};

use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NodeId(u32);
impl NodeId {
    pub fn root() -> Self {
        NodeId(0)
    }
    pub fn index(&self) -> usize {
        self.0 as usize
    }
    pub fn new(id: u32) -> Self {
        NodeId(id)
    }
}
impl Index<NodeId> for Vec<Node> {
    type Output = Node;
    fn index(&self, id: NodeId) -> &Node {
        &self[id.0 as usize]
    }
}
impl IndexMut<NodeId> for Vec<Node> {
    fn index_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self[id.0 as usize]
    }
}
