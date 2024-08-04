import os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init()

# List of colors for rainbow effect
rainbow_colors = [
    Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA
]

def get_image_files(folder):
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in extensions]
    return image_files

def create_pdf(image_files, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    for i, image_file in enumerate(tqdm(image_files, desc="Processing images", ncols=100)):
        color = rainbow_colors[i % len(rainbow_colors)]
        tqdm.write(color + f"Processing {image_file}" + Style.RESET_ALL)
        
        img = Image.open(image_file)
        img_width, img_height = img.size

        # Scale image to fit page
        aspect = img_width / float(img_height)
        if aspect > 1:
            # Landscape orientation
            scaled_width = width
            scaled_height = width / aspect
        else:
            # Portrait orientation
            scaled_width = height * aspect
            scaled_height = height

        # Center the image
        x = (width - scaled_width) / 2
        y = (height - scaled_height) / 2

        c.drawImage(image_file, x, y, scaled_width, scaled_height)
        c.showPage()

    c.save()

def main():
    folder = "C:\\Users\\Millind\\OneDrive\\Pictures\\AI"
    output_path = 'output.pdf'
    image_files = get_image_files(folder)
    
    if not image_files:
        print("No images found in the specified folder.")
        return

    create_pdf(image_files, output_path)
    print(f"PDF created successfully: {output_path}")

if __name__ == "__main__":
    main()
