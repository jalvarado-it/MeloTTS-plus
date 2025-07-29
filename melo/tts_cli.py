import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional, List
from melo.api import TTS
import  download_utils

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / 'tts.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Idiomas soportados
SUPPORTED_LANGUAGES = ['EN', 'ES', 'FR', 'ZH', 'JP', 'KR']

def validate_inputs(ckpt_path: str, text: str, language: str, output_dir: str) -> None:
    """Valida todos los parámetros de entrada"""
    # Validar checkpoint path
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise click.BadParameter(f"Checkpoint file not found: {ckpt_path}")
    
    # Validar texto
    if not text or not text.strip():
        raise click.BadParameter("Text cannot be empty")
    
    if len(text) > 10000:  # Límite razonable
        raise click.BadParameter("Text is too long (max 10,000 characters)")
    
    # Validar idioma
    if language.upper() not in SUPPORTED_LANGUAGES:
        raise click.BadParameter(f"Language '{language}' not supported. Available: {', '.join(SUPPORTED_LANGUAGES)}")
    
    # Validar directorio de salida
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise click.BadParameter(f"No permission to create directory: {output_dir}")
    
def load_text_from_file(file_path: str) -> str:
    """Carga texto desde un archivo"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Text loaded from file: {file_path}")
        return content
    except Exception as e:
        raise click.BadParameter(f"Error reading file {file_path}: {str(e)}")

def initialize_model(ckpt_path: str, language: str, device: str) -> TTS:
    """Inicializa el modelo TTS con manejo de errores"""
    try:
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using default")
            config_path = None
        
        logger.info(f"Loading model for language: {language}")
        logger.info(f"Device: {device}")
        
        model = TTS(
            language=language, 
            config_path=config_path, 
            ckpt_path=ckpt_path,
            device=device
        )
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise click.ClickException(f"Model loading failed: {str(e)}")
    
def generate_audio_files(model: TTS, text: str, output_dir: str, 
                        speakers: Optional[List[str]] = None,
                        speed: float = 1.0,
                        audio_format: str = 'wav') -> None:
    """Genera archivos de audio para cada speaker"""
    
    available_speakers = model.hps.data.spk2id.items()
    total_speakers = len(list(available_speakers))
    
    logger.info(f"Generating audio for {total_speakers} speakers")
    
    with click.progressbar(
        available_speakers, 
        label='Generating audio files',
        length=total_speakers
    ) as progress:
        
        for spk_name, spk_id in progress:
            # Filtrar speakers si se especificaron
            if speakers and spk_name not in speakers:
                continue
                
            try:
                # Crear directorio específico para cada speaker
                speaker_dir = Path(output_dir) / spk_name
                speaker_dir.mkdir(parents=True, exist_ok=True)
                
                # Generar nombre de archivo con timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"output_{timestamp}.{audio_format}"
                save_path = speaker_dir / filename
                
                # Generar audio
                model.tts_to_file(
                    text=text, 
                    speaker_id=spk_id, 
                    output_path=str(save_path),
                    speed=speed,
                    format=audio_format,
                    quiet=True
                )
                
                logger.info(f"Audio generated for {spk_name}: {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate audio for speaker {spk_name}: {str(e)}")
                continue
        
@click.command()
@click.option('--ckpt_path', '-m', 
              type=click.Path(exists=True), 
              required=True,
              defaut="checkpoints/"
              help="Path to the checkpoint file")
@click.option('--text', '-t', 
              type=str, 
              help="Text to speak (mutually exclusive with --text-file)")
@click.option('--text-file', '-f', 
              type=click.Path(exists=True),
              help="Path to text file (mutually exclusive with --text)")
@click.option('--language', '-l', 
              type=click.Choice(SUPPORTED_LANGUAGES, case_sensitive=False),
              default="EN",
              help="Language of the model")
@click.option('--output-dir', '-o', 
              type=click.Path(),
              default="outputs",
              help="Path to the output directory")
@click.option('--speakers', '-s',
              multiple=True,
              help="Specific speaker names to use (can be used multiple times)")
@click.option('--device', '-d',
              type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              default='auto',
              help="Device to use for inference")
@click.option('--speed',
              type=click.FloatRange(0.1, 3.0),
              default=1.0,
              help="Speech speed (0.1 to 3.0)")
@click.option('--format',
              type=click.Choice(['wav', 'mp3', 'flac']),
              default='wav',
              help="Output audio format")
@click.option('--list-speakers', 
              is_flag=True,
              help="List available speakers and exit")
@click.option('--verbose', '-v',
              is_flag=True,
              help="Enable verbose logging")
def main(ckpt_path: str, text: Optional[str], text_file: Optional[str], 
         language: str, output_dir: str, speakers: tuple, device: str,
         speed: float, format: str, list_speakers: bool, verbose: bool):
    """
    Advanced Text-to-Speech CLI tool using MeloTTS+
    
    Examples:
    \b
    # Generate from text
    python tts_cli.py -m checkpoint.pth -t "Hello world" -l EN
    
    # Generate from file
    python tts_cli.py -m checkpoint.pth -f input.txt -l EN
    
    # Use specific speakers
    python tts_cli.py -m checkpoint.pth -t "Hello" -s speaker1 -s speaker2
    
    # List available speakers
    python tts_cli.py -m checkpoint.pth --list-speakers
    """
    
    # Configurar nivel de logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validar que solo se use text o text-file
        if text and text_file:
            raise click.BadParameter("Cannot use both --text and --text-file")
        
        if not text and not text_file and not list_speakers:
            raise click.BadParameter("Must provide either --text, --text-file, or --list-speakers")
        
        # Inicializar modelo
        model = initialize_model(ckpt_path, language.upper(), device)
        
        # Listar speakers si se solicita
        if list_speakers:
            click.echo("Available speakers:")
            for spk_name, spk_id in model.hps.data.spk2id.items():
                click.echo(f"  - {spk_name} (ID: {spk_id})")
            return
        
        # Obtener texto
        if text_file:
            text = load_text_from_file(text_file)
        
        # Validar inputs
        validate_inputs(ckpt_path, text, language, output_dir)
        
        # Convertir speakers tuple a list
        speaker_list = list(speakers) if speakers else None
        
        # Generar archivos de audio
        generate_audio_files(
            model=model,
            text=text,
            output_dir=output_dir,
            speakers=speaker_list,
            speed=speed,
            audio_format=format
        )
        
        click.echo(f"✅ Audio generation completed! Files saved to: {output_dir}")
        
    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise click.ClickException(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
