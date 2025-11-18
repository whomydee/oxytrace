import logging

import structlog

try:
    from .config import AppConfig
except ImportError:
    from config import AppConfig

config = AppConfig()
LOG_LEVEL = config.log_level
RUNTIME_ENVIRONMENT = config.runtime_environment


shared_processors: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.stdlib.ExtraAdder(),
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.TimeStamper(fmt="iso"),  # datadog log pipeline remaps timestamp as datetime attribute
    structlog.processors.CallsiteParameterAdder(
        {
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.LINENO,
            structlog.processors.CallsiteParameter.PATHNAME,
        }
    ),
    # "event" is an internal keyword for message in structlog, if you want to log event attribute, use _event.
    # But you can't use message attribute, the log message will be converted to message attribute.
    structlog.processors.EventRenamer("message", "_event"),  # Should be the last in chain
]

log_renderer: list[structlog.types.Processor] = (
    [structlog.dev.ConsoleRenderer(event_key="message")]
    if RUNTIME_ENVIRONMENT == "local"  # type: ignore
    else [
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]
)

formatter = structlog.stdlib.ProcessorFormatter(
    # These run ONLY on `logging` entries that do NOT originate within
    # structlog.
    foreign_pre_chain=shared_processors,
    # These run on ALL entries after the pre_chain is done.
    processors=[
        # Remove _record & _from_structlog.
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        *log_renderer,
    ],
)

structlog.configure(
    # See https://www.structlog.org/en/stable/performance.html for increasing performance in production
    processors=shared_processors
    + [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Use OUR `ProcessorFormatter` to format all `logging` entries.
root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(LOG_LEVEL)

LOGGER = structlog.stdlib.get_logger("default_logger")
