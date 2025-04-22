# Add near the top of your script if not present
import base64
from pathlib import Path # Ensure Path is imported if used implicitly before
# ---

# ... inside the main_app function ...

# --- Display Chat Messages (with Response Parsing) ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # ... (user message handling) ...
        elif message["role"] == "assistant":
            response_content = message["content"]
            # ... (logging) ...

            # --- Response Parsing and Rendering Logic ---
            last_end = 0
            try:
                image_pattern = r'\[Image:\s*(.*?)\s*\]'
                for match in re.finditer(image_pattern, response_content):
                    start, end = match.span()
                    image_path_str = match.group(1).strip() # Extract the path string
                    image_path = Path(image_path_str) # Convert to Path object for easier handling

                    # Display text segment before the image marker
                    if start > last_end:
                        st.markdown(response_content[last_end:start], unsafe_allow_html=True)

                    # Display the image itself (using whichever method you settled on)
                    logger.debug(f"Found image marker, attempting to display: {image_path}")
                    if image_path.is_file():
                        # --- Display the inline image (Example using basic st.image) ---
                        # Replace this with your chosen display method (e.g., larger width or PIL resize)
                        st.image(str(image_path), width=400)
                        # --- End Inline Image Display ---


                        # --- ADDITION: Add Download Link ---
                        try:
                            # Get filename for the download attribute
                            filename = image_path.name

                            # Determine MIME type from file extension for Data URL robustness
                            ext = image_path.suffix.lstrip('.').lower()
                            # Add more image types if needed
                            known_image_types = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'svg']
                            if ext in known_image_types:
                                mime_type = f"image/{ext}"
                                # Special case for svg
                                if ext == 'svg': mime_type = 'image/svg+xml'
                            else:
                                # Fallback for unknown types or if no extension
                                mime_type = 'application/octet-stream'
                                logger.warning(f"Unknown image extension '{ext}' for {filename}. Using generic MIME type.")

                            # Read image bytes
                            with open(image_path, "rb") as f:
                                img_bytes = f.read()
                            # Encode to Base64
                            b64_encoded = base64.b64encode(img_bytes).decode()
                            # Create Data URL
                            data_url = f"data:{mime_type};base64,{b64_encoded}"

                            # Create and display the download link using markdown with HTML
                            # Using "Download Image" as link text. Customize style as needed.
                            link_text = "Download Image"
                            link_html = f'<a href="{data_url}" download="{filename}" style="display: inline-block; margin-top: 5px; text-decoration: underline; color: #1E88E5; font-size: 0.9em;">{link_text}</a>'
                            st.markdown(link_html, unsafe_allow_html=True)

                        except Exception as dl_err:
                            logger.error(f"Error creating download link for {image_path}: {dl_err}", exc_info=True)
                            st.caption("_(Could not create download link)_")
                        # --- END ADDITION ---

                    elif image_path_str: # Check if path string exists but file doesn't
                        st.error(f"Image not found at specified path: {image_path_str}")
                    else:
                        st.error("Invalid image marker found in response.")

                    last_end = end

                # Display any remaining text after the last image marker
                if last_end < len(response_content):
                    st.markdown(response_content[last_end:], unsafe_allow_html=True)

            except Exception as render_err:
                # ... (rest of the error handling remains the same) ...
