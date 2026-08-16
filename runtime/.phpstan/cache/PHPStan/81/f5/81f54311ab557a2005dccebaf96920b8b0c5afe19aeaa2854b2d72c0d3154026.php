<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/DataType.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\DataType
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-90ef7363dc53024652db34f112216e893a420d3ee2dccd443c284194f3e7490f',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\DataType',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/DataType.php',
      ),
    ),
    'namespace' => 'Rubix\\ML',
    'name' => 'Rubix\\ML\\DataType',
    'shortName' => 'DataType',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Data Type
 *
 * A high-level data type value object.
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 23,
    'endLine' => 244,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Stringable',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
      'CONTINUOUS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'CONTINUOUS',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '1',
          'attributes' => 
          array (
            'startLine' => 30,
            'endLine' => 30,
            'startTokenPos' => 58,
            'startFilePos' => 472,
            'endTokenPos' => 58,
            'endFilePos' => 472,
          ),
        ),
        'docComment' => '/**
 * The continuous data type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 30,
        'endLine' => 30,
        'startColumn' => 5,
        'endColumn' => 32,
      ),
      'CATEGORICAL' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'CATEGORICAL',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '2',
          'attributes' => 
          array (
            'startLine' => 37,
            'endLine' => 37,
            'startTokenPos' => 71,
            'startFilePos' => 585,
            'endTokenPos' => 71,
            'endFilePos' => 585,
          ),
        ),
        'docComment' => '/**
 * The categorical data type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 33,
      ),
      'IMAGE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'IMAGE',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '3',
          'attributes' => 
          array (
            'startLine' => 44,
            'endLine' => 44,
            'startTokenPos' => 84,
            'startFilePos' => 686,
            'endTokenPos' => 84,
            'endFilePos' => 686,
          ),
        ),
        'docComment' => '/**
 * The image data type code.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 44,
        'endLine' => 44,
        'startColumn' => 5,
        'endColumn' => 27,
      ),
      'OTHER' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'OTHER',
        'modifiers' => 1,
        'type' => NULL,
        'value' => 
        array (
          'code' => '0',
          'attributes' => 
          array (
            'startLine' => 51,
            'endLine' => 51,
            'startTokenPos' => 97,
            'startFilePos' => 813,
            'endTokenPos' => 97,
            'endFilePos' => 813,
          ),
        ),
        'docComment' => '/**
 * Any other data type that is not supported natively.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 51,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 27,
      ),
      'TYPE_STRINGS' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'TYPE_STRINGS',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[self::OTHER => \'other\', self::CONTINUOUS => \'continuous\', self::CATEGORICAL => \'categorical\', self::IMAGE => \'image\']',
          'attributes' => 
          array (
            'startLine' => 58,
            'endLine' => 63,
            'startTokenPos' => 110,
            'startFilePos' => 983,
            'endTokenPos' => 148,
            'endFilePos' => 1139,
          ),
        ),
        'docComment' => '/**
 * The human-readable string representations of the high-level data types.
 *
 * @var literal-string[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 58,
        'endLine' => 63,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
      'ALL' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'ALL',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '[self::CONTINUOUS, self::CATEGORICAL, self::IMAGE, self::OTHER]',
          'attributes' => 
          array (
            'startLine' => 70,
            'endLine' => 75,
            'startTokenPos' => 161,
            'startFilePos' => 1269,
            'endTokenPos' => 183,
            'endFilePos' => 1370,
          ),
        ),
        'docComment' => '/**
 * An array of all the high-level data type codes.
 *
 * @var list<int>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 70,
        'endLine' => 75,
        'startColumn' => 5,
        'endColumn' => 6,
      ),
    ),
    'immediateProperties' => 
    array (
      'code' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'name' => 'code',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The integer-encoded data type.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 82,
        'endLine' => 82,
        'startColumn' => 5,
        'endColumn' => 24,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
    ),
    'immediateMethods' => 
    array (
      'build' => 
      array (
        'name' => 'build',
        'parameters' => 
        array (
          'code' => 
          array (
            'name' => 'code',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 89,
            'endLine' => 89,
            'startColumn' => 34,
            'endColumn' => 42,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a new data type object.
 *
 * @param int $code
 */',
        'startLine' => 89,
        'endLine' => 92,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'detect' => 
      array (
        'name' => 'detect',
        'parameters' => 
        array (
          'value' => 
          array (
            'name' => 'value',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'mixed',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 100,
            'endLine' => 100,
            'startColumn' => 35,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a data type object from an example value.
 *
 * @param mixed $value
 * @return self
 */',
        'startLine' => 100,
        'endLine' => 120,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'continuous' => 
      array (
        'name' => 'continuous',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a continuous data type.
 *
 * @return self
 */',
        'startLine' => 127,
        'endLine' => 130,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'categorical' => 
      array (
        'name' => 'categorical',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a categorical data type.
 *
 * @return self
 */',
        'startLine' => 137,
        'endLine' => 140,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'image' => 
      array (
        'name' => 'image',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build an image data type.
 *
 * @return self
 */',
        'startLine' => 147,
        'endLine' => 150,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'other' => 
      array (
        'name' => 'other',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build an other data type.
 *
 * @return self
 */',
        'startLine' => 157,
        'endLine' => 160,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'all' => 
      array (
        'name' => 'all',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return an array with all possible data types.
 *
 * @return list<self>
 */',
        'startLine' => 167,
        'endLine' => 170,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'code' => 
          array (
            'name' => 'code',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 176,
            'endLine' => 176,
            'startColumn' => 33,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param int $code
 * @throws InvalidArgumentException
 */',
        'startLine' => 176,
        'endLine' => 183,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'code' => 
      array (
        'name' => 'code',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'int',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the integer-encoded data type.
 *
 * @return int
 */',
        'startLine' => 190,
        'endLine' => 193,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'isContinuous' => 
      array (
        'name' => 'isContinuous',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Is the data type continuous?
 *
 * @return bool
 */',
        'startLine' => 200,
        'endLine' => 203,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'isCategorical' => 
      array (
        'name' => 'isCategorical',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Is the data type categorical?
 *
 * @return bool
 */',
        'startLine' => 210,
        'endLine' => 213,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'isImage' => 
      array (
        'name' => 'isImage',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Is the data type an image resource?
 *
 * @return bool
 */',
        'startLine' => 220,
        'endLine' => 223,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      'isOther' => 
      array (
        'name' => 'isOther',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Does the data not belong to any type?
 *
 * @return bool
 */',
        'startLine' => 230,
        'endLine' => 233,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
      '__toString' => 
      array (
        'name' => '__toString',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the data type as a string.
 *
 * @return string
 */',
        'startLine' => 240,
        'endLine' => 243,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML',
        'declaringClassName' => 'Rubix\\ML\\DataType',
        'implementingClassName' => 'Rubix\\ML\\DataType',
        'currentClassName' => 'Rubix\\ML\\DataType',
        'aliasName' => NULL,
      ),
    ),
    'traitsData' => 
    array (
      'aliases' => 
      array (
      ),
      'modifiers' => 
      array (
      ),
      'precedences' => 
      array (
      ),
      'hashes' => 
      array (
      ),
    ),
  ),
));