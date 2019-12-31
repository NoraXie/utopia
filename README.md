## 概况

* utils3: 常用方法函数包, 支持Python3
* requirements3.txt: 使用见[安装和使用], 包含所有Python3所需包以及版本


## 使用样例
请见 `utils` 和 `utils3` 中的demo文件。


## 文件夹命名和代码规范
* 当文件夹名和文件名为英文时，应全部为小写，单词间用 下划线 `_` 连接。比如：`model_monitoring`
* 代码规范用的是Google style的标准：
    - [R](https://google.github.io/styleguide/Rguide.xml)
    - [Python](https://google.github.io/styleguide/pyguide.html)

    [Naming Convention](https://google.github.io/styleguide/pyguide.html?showone=Naming#Naming)

    module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name.

    *Names to Avoid*
    - single character names except for counters or iterators
    - dashes (-) in any package/module name
    - `__double_leading_and_trailing_underscore__` names (reserved by Python)


    *Naming Convention*
    - "Internal" means internal to a module or protected or private within a class.
    - Prepending a single underscore (`_`) has some support for protecting module variables and functions (not included with `import * from`). Prepending a double underscore (`__`) to an instance variable or method effectively serves to make the variable or method private to its class (using name mangling).
    - Place related classes and top-level functions together in a module. Unlike Java, there is no need to limit yourself to one class per module.
    Use CapWords for class names, but `lower_with_under.py` for module names. Although there are many existing modules named CapWords.py, this is now discouraged because it's confusing when the module happens to be named after a class. ("wait -- did I write import StringIO or from StringIO import StringIO?")


Guidelines derived from Guido's Recommendations

|Type | Public | Internal |
|----|----+----|
|Packages |`lower_with_under` | |
|Modules |`lower_with_under` | `_lower_with_under` |
|Classes	| `CapWords` | `_CapWords`|
|Exceptions | `CapWords` | |
|Functions | `lower_with_under()` | `_lower_with_under()` |
|Global/Class Constants | `CAPS_WITH_UNDER` | `_CAPS_WITH_UNDER` |
|Global/Class Variables | `lower_with_under` | `_lower_with_under` |
|Instance Variables | `lower_with_under`	 | `_lower_with_under` (protected) or `__lower_with_under` (private)
|Method Names | `lower_with_under()` | `_lower_with_under()` (protected) or `__lower_with_under()` (private)
|Function/Method Parameters | `lower_with_under` | |
|Local Variables | `lower_with_under` | |

## 安装和使用

* 安装依赖包
`pip install -r requirements3.txt`

* 使用
Linux 系统是~/.bashrc, Mac OS系统是 ~/.bash_profile, 在这个文件里面加入
`export PYTHONPATH="/Users/pintec/Documents/repos/genie/:$PYTHONPATH"`
替换`/Users/pintec/Documents/repos/genie/`为本地的路径
更改save好这个文件后
`source ~/.bashrc`
或
`source ~/.bash_profile`
之后既可以正常import这个包里面的函数了

使用Python3时可以参照：
```
from utils3.data_io_utils import *
```
