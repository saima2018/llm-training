import jinja2


class BaseTemplate:
    """用于读取jinja模板，方便按json格式配置参数"""

    def __init__(self, template_path=None):
        assert template_path, "template_path must be given"
        with open(template_path, encoding="utf-8") as fp:
            template = fp.read()
        self.template = jinja2.Template(template)

    def format(self, **kwargs):
        return self.template.render(**kwargs)
