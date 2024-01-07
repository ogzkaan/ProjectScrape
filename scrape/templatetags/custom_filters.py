from django import template

register = template.Library()

@register.filter(name='strip_quotes')
def strip_quotes(value):
    """Remove quotes from a string."""
    return value.replace('"', '').replace("'", "")