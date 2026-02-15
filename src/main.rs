use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::rc::Rc;

pub type ValueRef = Rc<RefCell<Value>>;

/// Node in the autograd computation graph.
///
/// `Value` stores:
/// - `data`: scalar computed in the forward pass
/// - `grad`: d(loss)/d(this node), accumulated in backward pass
/// - `children`: input nodes used to compute this node
/// - `local_grads`: local derivatives d(this node)/d(child)
///
/// The name `Value` follows micrograd-style name.
#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub children: Vec<ValueRef>,
    pub local_grads: Vec<f64>,
}

impl Value {
    pub fn new(data: f64) -> ValueRef {
        Rc::new(RefCell::new(Self {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        }))
    }

    pub fn with_graph(data: f64, children: Vec<ValueRef>, local_grads: Vec<f64>) -> ValueRef {
        debug_assert_eq!(
            children.len(),
            local_grads.len(),
            "children and local_grads must have the same length"
        );

        Rc::new(RefCell::new(Self {
            data,
            grad: 0.0,
            children,
            local_grads,
        }))
    }

    pub fn add(lhs: &ValueRef, rhs: &ValueRef) -> ValueRef {
        let lhs_data = lhs.borrow().data;
        let rhs_data = rhs.borrow().data;
        Self::with_graph(
            lhs_data + rhs_data,
            vec![Rc::clone(lhs), Rc::clone(rhs)],
            vec![1.0, 1.0],
        )
    }

    pub fn mul(lhs: &ValueRef, rhs: &ValueRef) -> ValueRef {
        let lhs_data = lhs.borrow().data;
        let rhs_data = rhs.borrow().data;
        Self::with_graph(
            lhs_data * rhs_data,
            vec![Rc::clone(lhs), Rc::clone(rhs)],
            vec![rhs_data, lhs_data],
        )
    }

    pub fn powf(x: &ValueRef, exponent: f64) -> ValueRef {
        let x_data = x.borrow().data;
        let out_data = x_data.powf(exponent);
        let local_grad = exponent * x_data.powf(exponent - 1.0);
        Self::with_graph(out_data, vec![Rc::clone(x)], vec![local_grad])
    }

    pub fn log(x: &ValueRef) -> ValueRef {
        let x_data = x.borrow().data;
        let out_data = x_data.ln();
        let local_grad = 1.0 / x_data;
        Self::with_graph(out_data, vec![Rc::clone(x)], vec![local_grad])
    }

    pub fn exp(x: &ValueRef) -> ValueRef {
        let x_data = x.borrow().data;
        let out_data = x_data.exp();
        let local_grad = out_data;
        Self::with_graph(out_data, vec![Rc::clone(x)], vec![local_grad])
    }

    pub fn relu(x: &ValueRef) -> ValueRef {
        let x_data = x.borrow().data;
        let out_data = x_data.max(0.0);
        let local_grad = if x_data > 0.0 { 1.0 } else { 0.0 };
        Self::with_graph(out_data, vec![Rc::clone(x)], vec![local_grad])
    }

    /// Runs reverse-mode autodiff from `root` through the computation graph.
    ///
    /// Algorithm:
    /// 1. Build a topological ordering of all ancestors of `root` using DFS.
    /// 2. Zero gradients for all visited nodes.
    /// 3. Seed `root.grad = 1.0` because d(root)/d(root) = 1.
    /// 4. Traverse nodes in reverse topological order and apply chain rule:
    ///    `child.grad += local_grad * node.grad`.
    ///
    /// In this project, `root` is typically a scalar loss node produced by a
    /// forward pass. Calling `backward(&loss)` populates gradients on all
    /// leaf/intermediate nodes that contributed to that loss.
    pub fn backward(root: &ValueRef) {
        fn create_topological_order(
            v: &ValueRef,
            visited: &mut HashSet<usize>,
            topo: &mut Vec<ValueRef>,
        ) {
            let id = Rc::as_ptr(v) as usize;
            if visited.insert(id) {
                let children = v.borrow().children.clone();
                for child in children {
                    create_topological_order(&child, visited, topo);
                }
                topo.push(Rc::clone(v));
            }
        }

        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        create_topological_order(root, &mut visited, &mut topo);

        for node in &topo {
            node.borrow_mut().grad = 0.0;
        }
        root.borrow_mut().grad = 1.0;

        for node in topo.into_iter().rev() {
            let node_ref = node.borrow();
            let node_grad = node_ref.grad;
            let children = node_ref.children.clone();
            let local_grads = node_ref.local_grads.clone();
            drop(node_ref);

            for (child, local_grad) in children.into_iter().zip(local_grads.into_iter()) {
                child.borrow_mut().grad += local_grad * node_grad;
            }
        }
    }
}

pub type Matrix = Vec<Vec<ValueRef>>;

#[derive(Debug, Clone, Copy)]
pub struct Architecture {
    /// Embedding width (hidden size) for each token position.
    pub n_embd: usize,
    /// Number of attention heads per transformer layer.
    pub n_head: usize,
    /// Number of stacked transformer layers.
    pub n_layer: usize,
    /// Maximum sequence length the model attends over.
    pub block_size: usize,
}

impl Architecture {
    /// Width of a single attention head (`n_embd / n_head`).
    pub fn head_dim(self) -> usize {
        self.n_embd / self.n_head
    }
}

/// Learnable parameters for one transformer block.
#[derive(Debug)]
pub struct TransformerLayer {
    /// Query projection weights for self-attention.
    pub attn_wq: Matrix,
    /// Key projection weights for self-attention.
    pub attn_wk: Matrix,
    /// Value projection weights for self-attention.
    pub attn_wv: Matrix,
    /// Output projection weights after attention heads are combined.
    pub attn_wo: Matrix,
    /// First feed-forward projection (expands width to `4 * n_embd`).
    pub mlp_fc1: Matrix,
    /// Second feed-forward projection (projects back to `n_embd`).
    pub mlp_fc2: Matrix,
}

/// Full GPT-style model parameter container.
#[derive(Debug)]
pub struct Model {
    pub arch: Architecture,
    pub vocab_size: usize,
    /// Token embedding table (token id -> embedding vector).
    pub wte: Matrix,
    /// Positional embedding table (position index -> embedding vector).
    pub wpe: Matrix,
    /// Final projection to vocabulary logits.
    pub lm_head: Matrix,
    /// Stack of transformer layers.
    pub layers: Vec<TransformerLayer>,
}

impl Model {
    pub fn new(arch: Architecture, vocab_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_std(arch, vocab_size, 0.08)
    }

    pub fn new_with_std(
        arch: Architecture,
        vocab_size: usize,
        std: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, std)?;

        let wte = Self::random_matrix(vocab_size, arch.n_embd, &normal, &mut rng);
        let wpe = Self::random_matrix(arch.block_size, arch.n_embd, &normal, &mut rng);
        let lm_head = Self::random_matrix(vocab_size, arch.n_embd, &normal, &mut rng);

        let mut layers = Vec::with_capacity(arch.n_layer);
        for _ in 0..arch.n_layer {
            layers.push(TransformerLayer {
                attn_wq: Self::random_matrix(arch.n_embd, arch.n_embd, &normal, &mut rng),
                attn_wk: Self::random_matrix(arch.n_embd, arch.n_embd, &normal, &mut rng),
                attn_wv: Self::random_matrix(arch.n_embd, arch.n_embd, &normal, &mut rng),
                attn_wo: Self::random_matrix(arch.n_embd, arch.n_embd, &normal, &mut rng),
                mlp_fc1: Self::random_matrix(4 * arch.n_embd, arch.n_embd, &normal, &mut rng),
                mlp_fc2: Self::random_matrix(arch.n_embd, 4 * arch.n_embd, &normal, &mut rng),
            });
        }

        Ok(Self {
            arch,
            vocab_size,
            wte,
            wpe,
            lm_head,
            layers,
        })
    }

    fn random_matrix<R: rand::Rng + ?Sized>(
        nout: usize,
        nin: usize,
        normal: &Normal<f64>,
        rng: &mut R,
    ) -> Matrix {
        (0..nout)
            .map(|_| {
                (0..nin)
                    .map(|_| Value::new(normal.sample(rng)))
                    .collect::<Vec<ValueRef>>()
            })
            .collect()
    }

    pub fn params(&self) -> Vec<ValueRef> {
        let mut params = Vec::new();
        Self::flatten_matrix(&self.wte, &mut params);
        Self::flatten_matrix(&self.wpe, &mut params);
        Self::flatten_matrix(&self.lm_head, &mut params);

        for layer in &self.layers {
            Self::flatten_matrix(&layer.attn_wq, &mut params);
            Self::flatten_matrix(&layer.attn_wk, &mut params);
            Self::flatten_matrix(&layer.attn_wv, &mut params);
            Self::flatten_matrix(&layer.attn_wo, &mut params);
            Self::flatten_matrix(&layer.mlp_fc1, &mut params);
            Self::flatten_matrix(&layer.mlp_fc2, &mut params);
        }

        params
    }

    fn flatten_matrix(matrix: &Matrix, out: &mut Vec<ValueRef>) {
        for row in matrix {
            for param in row {
                out.push(Rc::clone(param));
            }
        }
    }
}

fn get_input_dataset() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    const INPUT_FILE: &str = "input.txt";
    const NAMES_URL: &str =
        "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";

    if !Path::new(INPUT_FILE).exists() {
        let response = ureq::get(NAMES_URL).call()?;
        let body = response.into_string()?;
        fs::write(INPUT_FILE, body)?;
    }

    let content = fs::read_to_string(INPUT_FILE)?;
    let mut docs: Vec<String> = content
        .trim()
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();

    let mut rng = rand::thread_rng();
    docs.shuffle(&mut rng);
    Ok(docs)
}

fn get_unique_chars_tokenizer(docs: &[String]) -> Result<Vec<char>, Box<dyn std::error::Error>> {
    let mut uchars: Vec<char> = docs.iter().flat_map(|doc| doc.chars()).collect();

    uchars.sort();
    uchars.dedup();

    Ok(uchars)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docs = get_input_dataset()?;
    let uchars = get_unique_chars_tokenizer(&docs)?;
    let bos = uchars.len();
    let vocab_size = bos + 1;

    println!("num docs: {}", docs.len());
    println!("vocab size: {}", vocab_size);
    Ok(())
}
