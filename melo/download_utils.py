import torch
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Union
import utils
from cached_path import cached_path
from huggingface_hub import hf_hub_download

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

# URLs originales (mantenidas para compatibilidad)
DOWNLOAD_CKPT_URLS = {
    'EN': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth',
    'EN_V2': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/checkpoint.pth',
    'FR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/checkpoint.pth',
    'JP': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/checkpoint.pth',
    'ES': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/checkpoint.pth',
    'ZH': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth',
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth',
}

DOWNLOAD_CONFIG_URLS = {
    'EN': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/config.json',
    'EN_V2': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/config.json',
    'FR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/config.json',
    'JP': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/config.json',
    'ES': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/config.json',
    'ZH': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json',
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/config.json',
}

PRETRAINED_MODELS = {
    'G.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/G.pth',
    'D.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/D.pth',
    'DUR.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/DUR.pth',
}

LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}

# Nuevas funcionalidades sin romper compatibilidad
DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'melotts'
LOCAL_MODELS_DIR = Path('checkpoints')

class ModelNotFoundError(Exception):
    """Excepción personalizada para modelos no encontrados"""
    pass

def get_supported_languages() -> list:
    """Retorna lista de idiomas soportados"""
    return list(LANG_TO_HF_REPO_ID.keys())

def validate_language(language: str) -> str:
    """Valida y normaliza el idioma"""
    lang = language.upper()
    if lang not in LANG_TO_HF_REPO_ID:
        available = ', '.join(get_supported_languages())
        raise ValueError(f"Language '{language}' not supported. Available: {available}")
    return lang

def get_local_model_path(language: str, filename: str = "checkpoint.pth") -> Optional[Path]:
    """Busca modelo en directorios locales"""
    language = validate_language(language)
    
    # Buscar en directorio local estándar
    local_path = LOCAL_MODELS_DIR / language / filename
    if local_path.exists():
        logger.info(f"Found local model: {local_path}")
        return local_path
    
    # Buscar en cache
    cache_path = DEFAULT_CACHE_DIR / language / filename
    if cache_path.exists():
        logger.info(f"Found cached model: {cache_path}")
        return cache_path
    
    return None

def ensure_cache_dir(language: str) -> Path:
    """Asegura que el directorio de cache existe"""
    language = validate_language(language)
    cache_dir = DEFAULT_CACHE_DIR / language
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_with_progress(url: str, destination: Path, description: str = "Downloading") -> Path:
    """Descarga archivo con barra de progreso (placeholder para implementación futura)"""
    logger.info(f"{description}: {url}")
    try:
        # Por ahora usa cached_path, pero se puede extender con tqdm
        downloaded_path = cached_path(url)
        
        # Copiar a destino si es diferente
        if Path(downloaded_path) != destination:
            import shutil
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, destination)
            return destination
        
        return Path(downloaded_path)
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise ModelNotFoundError(f"Could not download model from {url}")

def load_or_download_config(locale: str, use_hf: bool = True, config_path: Optional[str] = None):
    """
    Función original mantenida para compatibilidad
    
    Args:
        locale: Idioma del modelo
        use_hf: Usar Hugging Face Hub
        config_path: Ruta personalizada al archivo de configuración
    
    Returns:
        Objeto de configuración (hparams)
    """
    if config_path is None:
        language = locale.split('-')[0].upper()
        
        # Primero buscar localmente
        local_config = get_local_model_path(language, "config.json")
        if local_config:
            config_path = str(local_config)
        else:
            # Descargar según método preferido
            if use_hf:
                if language not in LANG_TO_HF_REPO_ID:
                    raise ModelNotFoundError(f"Language {language} not available in HF Hub")
                
                logger.info(f"Downloading config from HF Hub for {language}")
                config_path = hf_hub_download(
                    repo_id=LANG_TO_HF_REPO_ID[language], 
                    filename="config.json",
                    cache_dir=str(DEFAULT_CACHE_DIR)
                )
            else:
                if language not in DOWNLOAD_CONFIG_URLS:
                    raise ModelNotFoundError(f"Language {language} not available via direct URL")
                
                logger.info(f"Downloading config from direct URL for {language}")
                config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])
    
    logger.info(f"Loading config from: {config_path}")
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale: str, device: str, use_hf: bool = True, ckpt_path: Optional[str] = None):
    """
    Función original mantenida para compatibilidad con mejoras
    
    Args:
        locale: Idioma del modelo
        device: Dispositivo de destino ('cpu', 'cuda', etc.)
        use_hf: Usar Hugging Face Hub
        ckpt_path: Ruta personalizada al checkpoint
    
    Returns:
        Estado del modelo cargado
    """
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        
        # Primero buscar localmente
        local_model = get_local_model_path(language, "checkpoint.pth")
        if local_model:
            ckpt_path = str(local_model)
        else:
            # Descargar según método preferido
            if use_hf:
                if language not in LANG_TO_HF_REPO_ID:
                    raise ModelNotFoundError(f"Language {language} not available in HF Hub")
                
                logger.info(f"Downloading model from HF Hub for {language}")
                ckpt_path = hf_hub_download(
                    repo_id=LANG_TO_HF_REPO_ID[language], 
                    filename="checkpoint.pth",
                    cache_dir=str(DEFAULT_CACHE_DIR)
                )
            else:
                if language not in DOWNLOAD_CKPT_URLS:
                    raise ModelNotFoundError(f"Language {language} not available via direct URL")
                
                logger.info(f"Downloading model from direct URL for {language}")
                ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])
    
    logger.info(f"Loading model from: {ckpt_path}")
    
    try:
        return torch.load(ckpt_path, map_location=device)
    except Exception as e:
        logger.error(f"Failed to load model from {ckpt_path}: {str(e)}")
        raise ModelNotFoundError(f"Could not load model: {str(e)}")

def load_pretrain_model():
    """Función original mantenida para compatibilidad"""
    logger.info("Loading pretrained models")
    return [cached_path(url) for url in PRETRAINED_MODELS.values()]

# Nuevas funciones avanzadas (extensiones no breaking)
def get_model_info(language: str) -> Dict[str, Union[str, bool]]:
    """Obtiene información sobre disponibilidad del modelo"""
    language = validate_language(language)
    
    local_model = get_local_model_path(language)
    local_config = get_local_model_path(language, "config.json")
    
    return {
        'language': language,
        'hf_repo': LANG_TO_HF_REPO_ID.get(language),
        'direct_url_available': language in DOWNLOAD_CKPT_URLS,
        'local_model_available': local_model is not None,
        'local_config_available': local_config is not None,
        'local_model_path': str(local_model) if local_model else None,
        'local_config_path': str(local_config) if local_config else None,
    }

def list_local_models() -> Dict[str, Dict[str, Optional[str]]]:
    """Lista todos los modelos disponibles localmente"""
    local_models = {}
    
    for language in get_supported_languages():
        model_path = get_local_model_path(language)
        config_path = get_local_model_path(language, "config.json")
        
        if model_path or config_path:
            local_models[language] = {
                'model_path': str(model_path) if model_path else None,
                'config_path': str(config_path) if config_path else None,
            }
    
    return local_models

def clear_cache(language: Optional[str] = None) -> None:
    """Limpia cache de modelos"""
    if language:
        language = validate_language(language)
        cache_dir = DEFAULT_CACHE_DIR / language
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache for {language}")
    else:
        if DEFAULT_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(DEFAULT_CACHE_DIR)
            logger.info("Cleared all model cache")

def setup_local_model(language: str, model_path: str, config_path: Optional[str] = None) -> None:
    """Configura modelo local en la estructura estándar"""
    language = validate_language(language)
    
    # Crear directorio local
    local_dir = LOCAL_MODELS_DIR / language
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar archivos
    import shutil
    
    # Copiar modelo
    model_src = Path(model_path)
    if not model_src.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_dst = local_dir / "checkpoint.pth"
    shutil.copy2(model_src, model_dst)
    logger.info(f"Model copied to: {model_dst}")
    
    # Copiar config si se proporciona
    if config_path:
        config_src = Path(config_path)
        if not config_src.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config_dst = local_dir / "config.json"
        shutil.copy2(config_src, config_dst)
        logger.info(f"Config copied to: {config_dst}")

# Funciones de utilidad adicionales
def get_cache_size() -> Dict[str, int]:
    """Obtiene el tamaño del cache por idioma"""
    sizes = {}
    
    if not DEFAULT_CACHE_DIR.exists():
        return sizes
    
    for lang_dir in DEFAULT_CACHE_DIR.iterdir():
        if lang_dir.is_dir():
            total_size = sum(f.stat().st_size for f in lang_dir.rglob('*') if f.is_file())
            sizes[lang_dir.name] = total_size
    
    return sizes

def verify_model_integrity(language: str) -> Dict[str, bool]:
    """Verifica integridad de modelos locales"""
    language = validate_language(language)
    
    results = {
        'model_exists': False,
        'config_exists': False,
        'model_loadable': False,
        'config_loadable': False,
    }
    
    # Verificar existencia
    model_path = get_local_model_path(language)
    config_path = get_local_model_path(language, "config.json")
    
    results['model_exists'] = model_path is not None
    results['config_exists'] = config_path is not None
    
    # Verificar carga
    if model_path:
        try:
            torch.load(str(model_path), map_location='cpu')
            results['model_loadable'] = True
        except Exception as e:
            logger.warning(f"Model not loadable: {e}")
    
    if config_path:
        try:
            utils.get_hparams_from_file(str(config_path))
            results['config_loadable'] = True
        except Exception as e:
            logger.warning(f"Config not loadable: {e}")
    
    return results