mod preprocessing;
mod modeling;

use std::fs::File;
use std::io::{Write, BufWriter};
use std::mem;
use std::time::Instant;
use sysinfo::{Pid, System, ProcessesToUpdate};
use winapi::shared::minwindef::FILETIME;
use winapi::um::processthreadsapi::GetProcessTimes;
use csv::Writer;

fn initialize_csv(sample: usize) -> Result<Writer<File>, Box<dyn std::error::Error>> {
    // Specify the output directory
    let output_dir = "../rust_metrics"; // Adjust this path as needed
    if !std::path::Path::new(output_dir).exists() {
        std::fs::create_dir_all(output_dir)?; // Create the directory if it doesn't exist
    }

    // Create the file in the output directory
    let filepath = format!("{}/N{}_metrics.csv", output_dir, sample);
    let writer = Writer::from_path(filepath)?;
    Ok(writer)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Data Analysis Pipeline");

    // Number of iterations
    let iterations = 5;
    let datasets = 100;

    for sample in [100, 250, 500, 750, 1000] {
        // Initialize a new CSV file for each sample
        let mut writer = initialize_csv(sample)?;
        writer.write_record(&["Iteration", "Dataset", "Step", "Time (s)", "Memory (MB)", "CPU Usage (%)", "Topics"])?;

        for i in 0..iterations {
            for j in 0..datasets {
                println!("\nIteration {}/{}", i + 1, iterations);
                println!("Dataset {}/{}", j + 1, datasets);
                println!("========================================");

                println!("Starting Data Analysis Pipeline");
                measure_step(
                    "preprocessing",
                    i + 1,
                    sample,
                    j + 1,
                    || preprocessing::start(&format!("../bootstrap_samples/N_{}/sample_{}", sample, j + 1)),
                    &mut writer,
                )?;
                measure_step("modeling", i + 1, sample, j + 1, || modeling::start(), &mut writer)?;
            }
        }
    }
    Ok(())
}

fn measure_step<F>(
    name: &str,
    iteration: usize,
    sample: usize,
    dataset: usize,
    step: F,
    writer: &mut Writer<File>,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<Vec<String>, Box<dyn std::error::Error>>,
{
    println!("Starting {} pipeline...", name);

    let timer = Instant::now();
    let mut sys = System::new_all();
    let pid = Pid::from(std::process::id() as usize);
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let memory_before = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
    let process_handle = unsafe { winapi::um::processthreadsapi::GetCurrentProcess() };
    let start_cpu_time = get_process_cpu_time(process_handle)?;

    let result = step();

    let elapsed = timer.elapsed();
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let memory_after = sys.process(pid).map(|p| p.memory()).unwrap_or(0);

    let memory_usage_b = memory_after;
    let memory_usage_mb = memory_usage_b as f64 / (1024.0*1024.0);

    let end_cpu_time = get_process_cpu_time(process_handle)?;
    let cpu_usage = calculate_cpu_usage(start_cpu_time, end_cpu_time, elapsed);

    println!("{} Metrics:", name);
    println!("  Time: {:.2?}", elapsed);
    println!("  Memory: {:.2} MB", memory_usage_mb);
    println!("  CPU Usage: {:.1}%", cpu_usage);
    println!();

    // Handle the result based on the step
    let topics = match result {
        Ok(topics) if name == "modeling" => topics.join(" | "), // Join topics for modeling
        Ok(_) => "N/A".to_string(), // Use "N/A" for preprocessing
        Err(e) => return Err(e),    // Propagate errors
    };

    // Write metrics to the CSV file
    writer.serialize((
        iteration,
        dataset,
        name,
        elapsed.as_secs_f64(),
        memory_usage_mb,
        cpu_usage,
        topics,
    ))?;
    writer.flush()?;

    Ok(())
}

// Windows-specific CPU time functions
fn get_process_cpu_time(handle: winapi::um::winnt::HANDLE) -> Result<u64, Box<dyn std::error::Error>> {
    unsafe {
        let mut creation_time: FILETIME = mem::zeroed();
        let mut exit_time: FILETIME = mem::zeroed();
        let mut kernel_time: FILETIME = mem::zeroed();
        let mut user_time: FILETIME = mem::zeroed();
        
        if GetProcessTimes(
            handle,
            &mut creation_time,
            &mut exit_time,
            &mut kernel_time,
            &mut user_time,
        ) == 0
        {
            return Err("Failed to get process times".into());
        }

        let total_time = file_time_to_u64(kernel_time) + file_time_to_u64(user_time);
        Ok(total_time)
    }
}

fn file_time_to_u64(ft: FILETIME) -> u64 {
    ((ft.dwHighDateTime as u64) << 32) | (ft.dwLowDateTime as u64)
}

fn calculate_cpu_usage(start: u64, end: u64, elapsed: std::time::Duration) -> f64 {
    let cpu_time_diff = end - start;
    let elapsed_ns = elapsed.as_nanos() as f64;
    let cpu_time_ns = cpu_time_diff as f64 * 100.0; // Convert 100ns units to ns
    
    (cpu_time_ns / elapsed_ns * 100.0).min(100.0)
}