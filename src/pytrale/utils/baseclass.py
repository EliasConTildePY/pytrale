from dataclasses import _MISSING_TYPE, fields


class DefaultDataClass:
    def __post_init__(self):
        # Loop through the fields to fill default values. Fields may live on
        # a frozen dataclass, so bypass __setattr__ via object.__setattr__.
        for field in fields(self):
            if (
                not isinstance(field.default, _MISSING_TYPE) and
                getattr(self, field.name) is None
            ):
                object.__setattr__(self, field.name, field.default)
