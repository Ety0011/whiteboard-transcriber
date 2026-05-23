import os

from playwright.sync_api import sync_playwright


def generate_pdf():
    # Get absolute path to your HTML file
    html_path = os.path.abspath("poster.html")

    with sync_playwright() as p:
        # Launch a headless browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to your local HTML file
        page.goto(f"file://{html_path}")

        # Wait for the network to be idle and the drawLifecycle JS to run
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(500)  # Extra buffer for SVG calculations

        # Render the PDF with exact A0 specifications
        page.pdf(
            path="poster.pdf",
            width="841mm",
            height="1189mm",
            print_background=True,  # Ensures background colors are preserved
            margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
        )

        browser.close()
    print("Success: A0 PDF generated as 'poster.pdf'")


if __name__ == "__main__":
    generate_pdf()
