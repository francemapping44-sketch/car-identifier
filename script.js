import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

const statusText = document.getElementById('status');
const imageUpload = document.getElementById('imageUpload');
const uploadLabel = document.getElementById('uploadLabel');
const imagePreview = document.getElementById('imagePreview');
const resultText = document.getElementById('result');

let classifier;

// Zero-Shot classification requires a target vocabulary.
// Expand this array with ANY specific make/model you want the AI to identify.
const carModels = [
    "Rover 200", "Ford Mustang", "Honda Civic", "Toyota Camry", 
    "Chevrolet Corvette", "Tesla Model 3", "Jeep Wrangler", 
    "Porsche 911", "BMW 3 Series", "Audi A4", "Subaru Outback const carModels = [
    // --- TOYOTA ---
    "Toyota Camry", "Toyota Corolla", "Toyota RAV4", "Toyota Prius", "Toyota Highlander", 
    "Toyota Tacoma", "Toyota Tundra", "Toyota 4Runner", "Toyota Supra", "Toyota Sienna", 
    "Toyota Land Cruiser", "Toyota GR86", "Toyota Hilux",

    // --- HONDA ---
    "Honda Civic", "Honda Accord", "Honda CR-V", "Honda Pilot", "Honda Odyssey", 
    "Honda HR-V", "Honda Passport", "Honda Ridgeline", "Honda Prelude",

    // --- FORD ---
    "Ford F-150", "Ford Mustang", "Ford Explorer", "Ford Escape", "Ford Edge", 
    "Ford Bronco", "Ford Ranger", "Ford Maverick", "Ford Focus", "Ford Fiesta", 
    "Ford Puma", //

    // --- CHEVROLET & GMC ---
    "Chevrolet Silverado", "Chevrolet Malibu", "Chevrolet Equinox", "Chevrolet Tahoe", 
    "Chevrolet Suburban", "Chevrolet Corvette", "Chevrolet Camaro", "Chevrolet Trailblazer", 
    "Chevrolet Traverse", "Chevrolet Colorado", "GMC Sierra", "GMC Yukon", "GMC Terrain",

    // --- NISSAN ---
    "Nissan Altima", "Nissan Sentra", "Nissan Maxima", "Nissan Rogue", "Nissan Pathfinder", 
    "Nissan Frontier", "Nissan Titan", "Nissan GT-R", "Nissan Leaf", "Nissan Murano", 
    "Nissan Qashqai", //
    "Nissan Juke", //

    // --- HYUNDAI & KIA ---
    "Hyundai Elantra", "Hyundai Sonata", "Hyundai Tucson", //
    "Hyundai Santa Fe", "Hyundai Palisade", "Hyundai Kona", "Hyundai Ioniq 5", "Hyundai Ioniq 6",
    "Kia Forte", "Kia K5", "Kia Soul", "Kia Sportage", //
    "Kia Sorento", "Kia Telluride", "Kia Stinger", "Kia EV6", "Kia EV9", "Kia Seltos",

    // --- BMW & MINI ---
    "BMW 3 Series", "BMW 5 Series", "BMW 7 Series", "BMW X3", "BMW X5", "BMW X7", 
    "BMW M3", "BMW M5", "BMW i4", "BMW iX", "BMW Z4", 
    "BMW iX3", //
    "MINI Cooper", //

    // --- MERCEDES-BENZ ---
    "Mercedes-Benz C-Class", "Mercedes-Benz E-Class", "Mercedes-Benz S-Class", 
    "Mercedes-Benz GLC", "Mercedes-Benz GLE", "Mercedes-Benz GLS", "Mercedes-Benz G-Class", 
    "Mercedes-Benz AMG GT",

    // --- AUDI ---
    "Audi A3", "Audi A4", "Audi A6", "Audi A8", "Audi Q3", "Audi Q5", "Audi Q7", 
    "Audi Q8", "Audi R8", "Audi e-tron",

    // --- VOLKSWAGEN ---
    "Volkswagen Jetta", "Volkswagen Golf", //
    "Volkswagen Passat", "Volkswagen Tiguan", //
    "Volkswagen Atlas", "Volkswagen ID.4", "Volkswagen GTI", "Volkswagen Polo",

    // --- SUBARU & MAZDA ---
    "Subaru Impreza", "Subaru Legacy", "Subaru Outback", "Subaru Forester", 
    "Subaru Crosstrek", "Subaru WRX", "Subaru BRZ", 
    "Mazda Mazda3", "Mazda Mazda6", "Mazda CX-30", "Mazda CX-5", "Mazda CX-50", 
    "Mazda CX-90", "Mazda MX-5 Miata",

    // --- JEEP, RAM & DODGE ---
    "Jeep Wrangler", "Jeep Grand Cherokee", "Jeep Cherokee", "Jeep Compass", 
    "Jeep Gladiator", 
    "Jeep Recon", //
    "Ram 1500", "Ram 2500",
    "Dodge Charger", "Dodge Challenger", "Dodge Durango", "Dodge Hornet",

    // --- TESLA ---
    "Tesla Model S", "Tesla Model 3", "Tesla Model X", "Tesla Model Y", "Tesla Cybertruck",

    // --- PORSCHE & LEXUS ---
    "Porsche 911", "Porsche Cayman", "Porsche Boxster", "Porsche Cayenne", 
    "Porsche Macan", "Porsche Panamera", "Porsche Taycan",
    "Lexus IS", "Lexus ES", "Lexus LS", "Lexus NX", "Lexus RX", "Lexus GX", "Lexus LX", "Lexus LC",

    // --- VOLVO & LAND ROVER ---
    "Volvo XC40", //
    "Volvo XC60", "Volvo XC90", "Volvo V60", "Volvo S60",
    "Land Rover Range Rover", "Land Rover Defender", "Land Rover Discovery", "Land Rover Range Rover Evoque",

    // --- EXOTICS, SUPER CARS & EV STARTUPS ---
    "Jaguar F-Type", 
    "Jaguar Type 00", //
    "Aston Martin Vantage", "Aston Martin DBX",
    "Ferrari SF90", "Ferrari Roma", "Ferrari 296 GTB", "Ferrari Purosangue", 
    "Ferrari Luce", //
    "Lamborghini Huracan", "Lamborghini Urus", "Lamborghini Revuelto",
    "McLaren 720S", "McLaren Artura", "McLaren 750S",
    "Rivian R1T", "Rivian S1S", 
    "Rivian R3X", //
    "Lucid Air", "Lucid Gravity",

    // --- GLOBAL TRENDING IMPORTS ---
    "Jaecoo 7", //
    "Vauxhall Corsa", //
    "Vauxhall Frontera", //
    "MG HS", //
    "MG ZS", "MG4",
    "BYD Atto 3", "BYD Han", "BYD Seal", 
    "BYD Sealion 7" //
];"
];

async function loadAI() {
    try {
        // Swap generic image classification for Zero-Shot text/image comparison
        classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch16');
        
        statusText.innerText = "AI Ready! Select a car image.";
        imageUpload.disabled = false;
        uploadLabel.classList.remove('disabled');
    } catch (error) {
        statusText.innerText = "Error loading AI architecture.";
        console.error(error);
    }
}

loadAI();

imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async function(e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
        resultText.innerText = "Analyzing make and model...";

        try {
            // Pass the image AND the custom vocabulary array to the AI
            const results = await classifier(e.target.result, carModels);
            
            // Extract highest confidence prediction
            const bestGuess = results[0].label;
            const confidence = Math.round(results[0].score * 100);
            
            resultText.innerText = `${confidence}% Match: ${bestGuess}`;
        } catch (error) {
            resultText.innerText = "Analysis failed. Try another image.";
        }
    }
    reader.readAsDataURL(file);
});
