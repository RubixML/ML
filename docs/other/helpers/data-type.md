# Data Type
Determine the data type of a variable according to Rubix ML's type system.

To determine the data type of a variable:
```php
public determine($variable) : int
```

> **Note**: The return value is an integer encoding of the datatype defined as constants on the DataType class.

Return true if the variable is categorical:
```php
public isCategorical($variable) : bool
```

Return true if the variable is categorical:
```php
public isContinuous($variable) : bool
```

Return true if the variable is a PHP resource:
```php
public isResource($variable) : bool
```

Return true if the variable is an unrecognized data type:
```php
public isOther($variable) : bool
```

### Example
```php
use Rubix\ML\Other\Helpers\DataType;

var_dump(DataType::determine('string'));

var_dump(DataType::isContinuous(16));

var_dump(DataType::isCategorical(18));
```

**Output:**

```sh
int(2) // Categorical
bool(true)
bool(false)
```