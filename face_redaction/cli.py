import pathlib
from typing import Optional

import typer
from rich import print
from rich.console import Console
from typing_extensions import Annotated

from face_redaction.processing import (
    FaceDetectionModel, 
    FaceRedactionStrategy, 
    MediaFileEditor,
)

app = typer.Typer(
    add_completion=False, # HINT: remove default --show-completion and --add-completion options
    help="Face detection and redaction tool")


@app.command()
def redact_faces(
    input_file: Annotated[str, typer.Argument()],
    output_file: Annotated[Optional[str], typer.Argument()] = None,
    face_detection_model: Optional[FaceDetectionModel] = typer.Option(
        FaceDetectionModel.default,
        "--face-detection-model", "-fd",
        help = "Face detection model"
    ),
    face_redaction_method: Optional[FaceRedactionStrategy] = typer.Option(
        FaceRedactionStrategy.blur,
        "--face-redaction-method", "-fr",
        help = "Face redaction method"
    ),
):
    """
    Redact faces in the a image or video file using the given redaction method (e.g. blur or solid color rectangle)
    """

    # Ensure valid input / output files
    editor = MediaFileEditor()
    file_path = pathlib.Path(input_file)
    if not file_path.is_file():
        typer.secho(
            f"Video file '{input_file}' not found. ",
            fg = typer.colors.RED,
        )
        raise typer.Exit(1)
    if not (
        editor.is_valid_image(input_file) or 
        editor.is_valid_video(input_file) ):        
        typer.secho(
            f"File '{input_file}' is not valid or supported image / video file. Check file formats supported using 'info' command",
            fg = typer.colors.RED,
        )
        raise typer.Exit(1)


    if output_file is None or output_file == input_file:
        output_file = f"{input_file}_redacted_{face_redaction_method}{file_path.suffix}"
    
    # Convert image / video files 
    console = Console()
    if editor.is_valid_image(input_file):
        print(f"Image file '{input_file}' will be updated and saved in '{output_file}'")

        editor.redact_faces_in_image(input_file=input_file,
            output_file=output_file,
            detection_model=face_detection_model,
            face_redaction_method=face_redaction_method) 


    if editor.is_valid_video(input_file):
        print(f"Video file '{input_file}' will be updated and saved in '{output_file}'")
        with console.status(f"[green] processing video conversion..") as status:

            editor.redact_faces_in_video(input_file=input_file,
                output_file=output_file,
                detection_model=face_detection_model,
                face_redaction_method=face_redaction_method)  
  
    console.print(f"Redacting faces Done!")


@app.command()
def redact_faces_stream(
    output_file: Annotated[Optional[str], typer.Argument()] = None,
    face_detection_model: Optional[FaceDetectionModel] = typer.Option(
        FaceDetectionModel.default,
        "--face-detection-model", "-fd",
        help = "Face detection model"
    ),
    face_redaction_method: Optional[FaceRedactionStrategy] = typer.Option(
        FaceRedactionStrategy.blur,
        "--face-redaction-method", "-fr",
        help = "Face redaction method"
    ),
    show_preview: Optional[bool] = typer.Option(
        True,
        "--show-preview", "-v",
        help = "Show video previewś"
    ),
):
    """
    Redact faces in the camera captured video
    """
    editor = MediaFileEditor()
    output_folder = pathlib.Path(output_file).parent
    if not output_folder.exists():
        typer.secho(
            f"Output file folder '{output_folder}' not found. ",
            fg = typer.colors.RED,
        )
        raise typer.Exit(1)
    if not (
        editor.is_valid_video(output_file) ):        
        typer.secho(
            f"File '{output_file}' is not valid or supported video file. Check file formats supported using 'info' command",
            fg = typer.colors.RED,
        )
        raise typer.Exit(1)
    console = Console()
    print(f"Video will be captured and file '{output_file}' will be saved.")

    with console.status(f"[green] processing video conversion. Press Ctrl+C to stop video capture.") as status:

        editor.redact_faces_in_stream(
            output_file=output_file,
            detection_model=face_detection_model,
            face_redaction_method=face_redaction_method,
            show_video_preview=show_preview,
        )
        
        console.print(f"Video capture and redacting faces Done!")
        

@app.command()
def info():
    """
    Shows information about the supported image and video file formats
    """

    editor = MediaFileEditor()
    print(
    f"Face redaction tool:\n"
    f"====================\n"    
    f"File formats supported:\n"
    f"Image: {editor.image_formats_supported}\n"
    f"Video: {editor.video_formats_supported}")


def main():
    """
    Main entry point
    """
    app()

if __name__ == "__main__":
    main()