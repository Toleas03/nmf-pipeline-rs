use anyhow::Result;
use csv::ReaderBuilder;
use serde::Deserialize;
use serde_json;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use std::error::Error;
use rand_distr::Uniform;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Deserialize)]
struct Record {
    index: usize,
    tokens: String,
}

fn load_documents(filepath: &str) -> Result<Vec<Vec<String>>> {
    let file = File::open(filepath)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut documents = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        let tokens: Vec<String> = serde_json::from_str(&record.tokens)?;
        documents.push(tokens);
    }
    Ok(documents)
}

fn build_vocabulary(documents: &[Vec<String>], min_df: usize) -> HashMap<String, usize> {
    let mut doc_counts = HashMap::new();
    for doc in documents {
        let unique_tokens: HashSet<_> = doc.iter().collect();
        for token in unique_tokens {
            *doc_counts.entry(token.clone()).or_insert(0) += 1;
        }
    }

    let mut vocab = HashMap::new();
    let mut next_idx = 0;
    for (token, count) in doc_counts {
        if count >= min_df {
            vocab.insert(token, next_idx);
            next_idx += 1;
        }
    }
    //println!("Vocab: {:?}", vocab);
    vocab
}

fn create_tfidf_matrix(documents: &[Vec<String>], vocab: &HashMap<String, usize>) -> Array2<f32> {
    let (num_docs, vocab_size) = (documents.len(), vocab.len());
    let mut tf = Array2::<f32>::zeros((num_docs, vocab_size));
    let mut idf = Array1::<f32>::zeros(vocab_size);

    // Calculate Term Frequency (TF) using filtered document length
    for (doc_idx, doc) in documents.iter().enumerate() {
        let mut valid_tokens = 0;
        for token in doc {
            if vocab.contains_key(token) {
                valid_tokens += 1;
            }
        }
        if valid_tokens == 0 { continue; }

        let doc_len = valid_tokens as f32;
        for token in doc {
            if let Some(&token_idx) = vocab.get(token) {
                tf[[doc_idx, token_idx]] += 1.0 / doc_len;
            }
        }
    }

    // Calculate IDF with smoothing to ensure positivity
    let num_docs_f32 = num_docs as f32;
    for (token, &token_idx) in vocab {
        let docs_with_token = documents.iter()
            .filter(|doc| doc.contains(token))
            .count() as f32;
        idf[token_idx] = 1.0 + ((num_docs_f32 + 1.0) / (docs_with_token + 1.0)).ln();
    }

    // Calculate TF-IDF and ensure non-negativity
    let mut tfidf = tf;
    for mut row in tfidf.rows_mut() {
        row *= &idf;
        row.mapv_inplace(|x| x.max(0.0));  // Clip negative values to 0
    }

    tfidf
}

fn nmf(v: &Array2<f32>, k: usize, max_iter: usize, tol: f32) -> (Array2<f32>, Array2<f32>) {
    let (docs, vocab_size) = v.dim();
    let eps = 1e-10;
    let lambda = 0.01;  // Reduced regularization

    // Initialize with higher values to prevent underflow
    let w_dist = Uniform::new(0.1, 1.0);
    let h_dist = Uniform::new(0.1, 1.0);
    let mut w = Array2::random((docs, k), w_dist);
    let mut h = Array2::random((k, vocab_size), h_dist);

    let mut error_at_init = 0 as f32;
    let mut prev_error = 0 as f32;

    for iter in 0..max_iter {
        // Update H with safer regularization
        let wt = w.t();
        let numerator_h = wt.dot(v);
        let denominator_h = wt.dot(&w.dot(&h)) + lambda + eps;
        h = h * &(numerator_h / denominator_h);

        // Update W with safer regularization
        let ht = &h.t();
        let numerator_w = v.dot(ht);
        let denominator_w = w.dot(&h).dot(ht) + lambda + eps;
        w = w * &(numerator_w / denominator_w);

        // Calculate the Frobenius norm
        let wh = w.dot(&h);
        let err = v - &wh;
        let error = err.mapv(|x| x.powi(2)).sum();
        
        
        if iter == 0 {
            error_at_init = error;
            prev_error = error_at_init;
            
        }

        let error_diff = (prev_error - error) / error_at_init;

        prev_error = error;

        if error_diff < tol && iter > 0 {
            // println!("Error {}, Prev {}, Diff {}, innit {}", error, prev_error, error_diff, error_at_init);
            // println!("Converged after {} iterations", iter + 1);
            break;
        }
        if iter % 10 == 0 {
            // println!("Error {}, Prev {}, Diff {}, innit {}", error, prev_error, error_diff, error_at_init);
            // println!("Iteration {}: error = {}", iter, error_diff);
        }
    }
    (w, h)
}


fn print_topics(h: &Array2<f32>, vocab: &HashMap<String, usize>) -> Vec<String> {
    let mut feature_names = vec![""; vocab.len()];
    for (word, &idx) in vocab {
        feature_names[idx] = word;
    }

    let mut topics = Vec::new();

    for (topic_idx, topic) in h.axis_iter(Axis(0)).enumerate() {
        let mut weights: Vec<(&&str, f32)> = feature_names.iter()
            .zip(topic.iter().copied())
            .collect();

        weights.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut topic_string = format!("Topic {}: ", topic_idx);
        for (word, weight) in weights.iter().take(10) {
            if *weight > 0.001 {
                topic_string.push_str(&format!("{} ", word));
            }
        }
        topics.push(topic_string.trim_end().to_string());
    }

    topics
}

fn save_topic_distributions(w: &Array2<f32>, output_path: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(output_path)?;

    // Create header: ["Document", "Topic0", "Topic1", ...]
    let num_topics = w.ncols();
    let mut headers = vec!["Document".to_string()];
    headers.extend((0..num_topics).map(|i| format!("Topic{}", i)));
    wtr.write_record(&headers)?;

    // Write each document's topic distribution
    for (doc_idx, topic_weights) in w.rows().into_iter().enumerate() {
        let mut record = vec![doc_idx.to_string()];
        record.extend(topic_weights.iter().map(|w| format!("{:.6}", w)));
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}

pub fn start() -> Result<Vec<String>, Box<dyn Error>> {
    let min_df = 3;
    let k = 5;
    let max_iter = 200;
    let tol = 1e-4;

    let documents = load_documents("tokens.csv")?;
    let vocab = build_vocabulary(&documents, min_df);
    let tfidf = create_tfidf_matrix(&documents, &vocab);

    let (w, h) = nmf(&tfidf, k, max_iter, tol);

    save_topic_distributions(&w, "document_topic_distributions.csv")?;
    let topics = print_topics(&h, &vocab);

    Ok(topics)
}