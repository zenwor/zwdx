import dill
import traceback
import base64
import torch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("client")

def safe_dumps(obj):
    return dill.dumps(obj, byref=False, recurse=True)

def safe_loads(bytes_obj):
    return dill.loads(bytes_obj)

def deserialize_model_payload(payload):
    if payload is None:
        return None
    
    if payload.get("class_source") and payload.get("state_dict"):
        try:
            import torch
            
            class_source = payload["class_source"]
            class_name = payload["class_name"]
            state_dict_bytes = payload["state_dict"]
            
            namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
            
            exec(class_source, namespace)
            
            model_class = namespace.get(class_name)
            if model_class is None:
                logger.error(f"[Deserialize Model] Class {class_name} not found in namespace")
                raise ValueError(f"Class {class_name} not found")
            
            model = model_class()
            
            state_dict = dill.loads(base64.b64decode(state_dict_bytes))
            model.load_state_dict(state_dict)
            
            logger.info(f"[Deserialize Model] Successfully reconstructed {class_name} from source")
            return model
        except Exception as e:
            logger.warning(f"[Deserialize Model] Source reconstruction failed: {e}")
            logger.warning(traceback.format_exc())
    
    if payload.get("pickled"):
        try:
            model = dill.loads(base64.b64decode(payload["pickled"]))
            logger.info(f"[Deserialize Model] Successfully loaded model from pickle")
            return model
        except Exception as e:
            logger.error(f"[Deserialize Model] Pickle deserialization failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    logger.error("[Deserialize Model] No valid source or pickle found in payload")
    return None

def deserialize_function_payload(payload):
    if payload is None:
        return None
    
    # Try source code reconstruction first
    if payload.get("source"):
        try:
            source = payload["source"]
            func_name = payload["name"]
            
            namespace = {}
            
            exec(source, namespace)
            
            func = namespace.get(func_name)
            if func:
                logger.info(f"[Deserialize] Successfully reconstructed {func_name} from source")
                return func
            else:
                logger.warning(f"[Deserialize] Function {func_name} not found in namespace after exec")
        except Exception as e:
            logger.warning(f"[Deserialize] Source reconstruction failed: {e}")
            logger.warning(traceback.format_exc())
    
    # Fallback to pickle
    if payload.get("pickled"):
        try:
            func = dill.loads(base64.b64decode(payload["pickled"]))
            logger.info(f"[Deserialize] Successfully loaded function from pickle")
            return func
        except Exception as e:
            logger.error(f"[Deserialize] Pickle deserialization failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    logger.error("[Deserialize] No valid source or pickle found in payload")
    return None

def rebuild_optimizer(model, optimizer_config):
    cls = getattr(torch.optim, optimizer_config["class"])
    return cls(model.parameters(), **optimizer_config["kwargs"])
