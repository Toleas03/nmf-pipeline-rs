use csv::Writer;
use regex::Regex;
use serde::ser;
use serde_json;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use stemmer::Stemmer;
use walkdir::WalkDir;   

#[derive(Debug, serde::Serialize)]
struct TextData {
    index: u32,
    tokens: String, // Store tokens as a formatted string
}

#[derive(Debug, serde::Serialize)]
struct FileData {
    index: u32,
    file_path: String,
}

fn load_stopwords(filepath: &str) -> Result<HashSet<String>, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let stopwords: HashSet<String> = reader.lines()
        .filter_map(Result::ok)
        .collect();
    Ok(stopwords)
}

fn preprocess_text(text: &str, stopwords: &HashSet<String>) -> Vec<String> {
    // Remove special characters and numbers
    let re = Regex::new(r"[^a-zA-Z\s]").unwrap();
    let cleaned = re.replace_all(text, " ").to_lowercase();

    // Tokenize and filter empty strings
    let tokens: Vec<String> = cleaned.split_whitespace()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !stopwords.contains(s))
        .collect();

    // Lemmatization (using stemming as a simple approximation)
    let mut stemmer = Stemmer::new("english").unwrap();
    let tokens = tokens.iter()
        .map(|word| stemmer.stem(word).to_string())
        .collect();

    tokens
}

fn process_files(input_path: &str, output_path: &str, files_csv: &str, stopwords: &HashSet<String>) -> Result<(), Box<dyn Error>> {
    let mut text_writer = Writer::from_path(output_path)?;
    let mut file_writer = Writer::from_path(files_csv)?;
    let mut index: u32 = 0;
    println!("Processing files in {}...", input_path);
    for entry in WalkDir::new(input_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "txt") {
            let content = std::fs::read_to_string(path)?;
            let tokens = preprocess_text(&content, stopwords);

            //let tokens_str = format!("[{}]", tokens.join(", ")); // Manually format tokens as a string
            let tokens_str = serde_json::to_string(&tokens)?; // Use serde_json to format tokens as a string
                                                              //let lemmatized_str = get_words_from_string(&tokens_str, "./lemmas.csv", "Vec");

            //println!(lemmatized_str);
            let text_data = TextData {
                index,
                tokens: tokens_str,
            };

            let file_data = FileData {
                index,
                file_path: path.to_string_lossy().into_owned(),
            };

            text_writer.serialize(&text_data)?;
            file_writer.serialize(&file_data)?;

            index += 1;
        }
    }

    text_writer.flush()?;
    file_writer.flush()?;
    Ok(())
}

pub fn start(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let tokens_csv = "tokens.csv";
    let files_csv = "files.csv";
    let stopwords_file = "../stopwords.txt";

    if Path::new(tokens_csv).exists() {
        std::fs::remove_file(tokens_csv)?;
    }
    if Path::new(files_csv).exists() {
        std::fs::remove_file(files_csv)?;
    }


    let stopwords = load_stopwords(stopwords_file)?;
    process_files(path, tokens_csv, files_csv, &stopwords)?;
    println!("Preprocessing completed for path: {}", path);

    // Return an empty Vec<String> to match the expected type
    Ok(Vec::new())
}
