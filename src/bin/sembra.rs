use clap::Parser;
use image::GenericImageView;
use sembra::{
    resize, ResizeConfig, EnergyMode, ResizeOrder,
    image_to_bool_mask
};

/// CLI for seam carving image resizing
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input image path
    #[arg(long)]
    input: String,

    /// Output image path
    #[arg(long)]
    output: String,

    /// Target width
    #[arg(long)]
    width: Option<usize>,

    /// Target height
    #[arg(long)]
    height: Option<usize>,

    /// Energy mode: "backward" or "forward"
    #[arg(long, default_value="backward")]
    energy_mode: String,

    /// Order mode: "width-first" or "height-first"
    #[arg(long, default_value="width-first")]
    order: String,

    /// Keep mask image path (optional)
    #[arg(long)]
    keep_mask: Option<String>,

    /// Drop mask image path (optional)
    #[arg(long)]
    drop_mask: Option<String>,

    /// Step ratio for expansions
    #[arg(long, default_value="0.5")]
    step_ratio: f32
}

fn main() {
    let cli = Cli::parse();

    // Load input image
    let input_img = image::open(&cli.input)
        .unwrap_or_else(|e| {
            eprintln!("Error: Failed to open input image '{}': {}", cli.input, e);
            std::process::exit(1);
        });

    let (width, height) = input_img.dimensions();
    println!("Loaded image: {}x{}", width, height);

    // Load optional masks
    let keep_mask = cli.keep_mask.as_ref().map(|path| {
        let km_img = image::open(path)
            .unwrap_or_else(|e| {
                eprintln!("Error: Failed to open keep mask '{}': {}", path, e);
                std::process::exit(1);
            });
        image_to_bool_mask(&km_img)
    });

    let drop_mask = cli.drop_mask.as_ref().map(|path| {
        let dm_img = image::open(path)
            .unwrap_or_else(|e| {
                eprintln!("Error: Failed to open drop mask '{}': {}", path, e);
                std::process::exit(1);
            });
        image_to_bool_mask(&dm_img)
    });

    // Parse energy mode
    let energy_mode = match cli.energy_mode.as_str() {
        "backward" => EnergyMode::Backward,
        "forward" => EnergyMode::Forward,
        _ => {
            eprintln!("Error: Invalid energy mode '{}'. Use 'backward' or 'forward'.", cli.energy_mode);
            std::process::exit(1);
        }
    };

    // Parse resize order
    let order = match cli.order.as_str() {
        "width-first" => ResizeOrder::WidthFirst,
        "height-first" => ResizeOrder::HeightFirst,
        _ => {
            eprintln!("Error: Invalid order '{}'. Use 'width-first' or 'height-first'.", cli.order);
            std::process::exit(1);
        }
    };

    // Build configuration
    let config = ResizeConfig {
        width: cli.width,
        height: cli.height,
        energy_mode,
        order,
        keep_mask,
        drop_mask,
        step_ratio: cli.step_ratio,
    };

    // Display resize info
    if let Some(w) = config.width {
        println!("Target width: {}", w);
    }
    if let Some(h) = config.height {
        println!("Target height: {}", h);
    }
    println!("Energy mode: {:?}", energy_mode);
    println!("Resize order: {:?}", order);

    // Perform seam carving
    println!("Processing...");
    let resized = resize(input_img, config)
        .unwrap_or_else(|e| {
            eprintln!("Error: Seam carving failed: {}", e);
            std::process::exit(1);
        });

    // Save result
    let (out_width, out_height) = resized.dimensions();
    resized.save(&cli.output)
        .unwrap_or_else(|e| {
            eprintln!("Error: Failed to save output image '{}': {}", cli.output, e);
            std::process::exit(1);
        });

    println!("Seam carving complete. Saved to {}", cli.output);
    println!("Output dimensions: {}x{}", out_width, out_height);
}
