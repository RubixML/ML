<?php declare(strict_types = 1);

// osfsl-/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Vector.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Tensor\Vector
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-b2a7da0874de7a151518add137c4b8b04544c2367c0c47ca8239a808f800a4f2-8.4-6.70.0.3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Tensor\\Vector',
        'filename' => '/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Vector.php',
      ),
    ),
    'namespace' => 'Tensor',
    'name' => 'Tensor\\Vector',
    'shortName' => 'Vector',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Vector
 *
 * A one dimensional (rank 1) tensor with integer and/or floating point elements.
 *
 * @category    Scientific Computing
 * @package     Rubix/Tensor
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 26,
    'endLine' => 2329,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Tensor\\Tensor',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'a' => 
      array (
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'name' => 'a',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The 1-d sequential array that holds the values of the vector.
 *
 * @var list<float>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 33,
        'endLine' => 33,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'n' => 
      array (
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'name' => 'n',
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
 * The number of elements in the vector.
 *
 * @var int<0,max>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 40,
        'endLine' => 40,
        'startColumn' => 5,
        'endColumn' => 21,
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
          'a' => 
          array (
            'name' => 'a',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 48,
                'endLine' => 48,
                'startTokenPos' => 113,
                'startFilePos' => 983,
                'endTokenPos' => 114,
                'endFilePos' => 984,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 48,
            'endLine' => 48,
            'startColumn' => 34,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Factory method to build a new vector from an array.
 *
 * @param (int|float)[] $a
 * @return mixed
 */',
        'startLine' => 48,
        'endLine' => 51,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'quick' => 
      array (
        'name' => 'quick',
        'parameters' => 
        array (
          'a' => 
          array (
            'name' => 'a',
            'default' => 
            array (
              'code' => '[]',
              'attributes' => 
              array (
                'startLine' => 59,
                'endLine' => 59,
                'startTokenPos' => 150,
                'startFilePos' => 1230,
                'endTokenPos' => 151,
                'endFilePos' => 1231,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 59,
            'endLine' => 59,
            'startColumn' => 34,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a vector foregoing any validation for quicker instantiation.
 *
 * @param (int|float)[] $a
 * @return mixed
 */',
        'startLine' => 59,
        'endLine' => 62,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'zeros' => 
      array (
        'name' => 'zeros',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 70,
            'endLine' => 70,
            'startColumn' => 34,
            'endColumn' => 39,
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
 * Build a vector of zeros with n elements.
 *
 * @param int $n
 * @return self
 */',
        'startLine' => 70,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'ones' => 
      array (
        'name' => 'ones',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 81,
            'endLine' => 81,
            'startColumn' => 33,
            'endColumn' => 38,
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
 * Build a vector of ones with n elements.
 *
 * @param int $n
 * @return self
 */',
        'startLine' => 81,
        'endLine' => 84,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'fill' => 
      array (
        'name' => 'fill',
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
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 94,
            'endLine' => 94,
            'startColumn' => 33,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 94,
            'endLine' => 94,
            'startColumn' => 47,
            'endColumn' => 52,
            'parameterIndex' => 1,
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
 * Fill a vector with a given value.
 *
 * @param float $value
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 94,
        'endLine' => 102,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'rand' => 
      array (
        'name' => 'rand',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 111,
            'endLine' => 111,
            'startColumn' => 33,
            'endColumn' => 38,
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
 * Return a random uniform vector with values between 0 and 1.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 111,
        'endLine' => 127,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'gaussian' => 
      array (
        'name' => 'gaussian',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 137,
            'endLine' => 137,
            'startColumn' => 37,
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
 * Return a standard normally distributed (Gaussian) random vector with mean 0
 * and unit variance.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 137,
        'endLine' => 162,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'poisson' => 
      array (
        'name' => 'poisson',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 172,
            'endLine' => 172,
            'startColumn' => 36,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'lambda' => 
          array (
            'name' => 'lambda',
            'default' => 
            array (
              'code' => '1.0',
              'attributes' => 
              array (
                'startLine' => 172,
                'endLine' => 172,
                'startTokenPos' => 662,
                'startFilePos' => 3784,
                'endTokenPos' => 662,
                'endFilePos' => 3786,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 172,
            'endLine' => 172,
            'startColumn' => 44,
            'endColumn' => 62,
            'parameterIndex' => 1,
            'isOptional' => true,
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
 * Generate a vector with n elements from a Poisson distribution.
 *
 * @param int $n
 * @param float $lambda
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 172,
        'endLine' => 199,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'uniform' => 
      array (
        'name' => 'uniform',
        'parameters' => 
        array (
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 208,
            'endLine' => 208,
            'startColumn' => 36,
            'endColumn' => 41,
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
 * Return a uniform random vector with mean 0 and unit variance.
 *
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 208,
        'endLine' => 224,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'range' => 
      array (
        'name' => 'range',
        'parameters' => 
        array (
          'start' => 
          array (
            'name' => 'start',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 234,
            'endLine' => 234,
            'startColumn' => 34,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'end' => 
          array (
            'name' => 'end',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 234,
            'endLine' => 234,
            'startColumn' => 48,
            'endColumn' => 57,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'interval' => 
          array (
            'name' => 'interval',
            'default' => 
            array (
              'code' => '1.0',
              'attributes' => 
              array (
                'startLine' => 234,
                'endLine' => 234,
                'startTokenPos' => 963,
                'startFilePos' => 5128,
                'endTokenPos' => 963,
                'endFilePos' => 5130,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 234,
            'endLine' => 234,
            'startColumn' => 60,
            'endColumn' => 80,
            'parameterIndex' => 2,
            'isOptional' => true,
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
 * Return evenly spaced values within a given interval.
 *
 * @param float $start
 * @param float $end
 * @param float $interval
 * @return self
 */',
        'startLine' => 234,
        'endLine' => 237,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'linspace' => 
      array (
        'name' => 'linspace',
        'parameters' => 
        array (
          'min' => 
          array (
            'name' => 'min',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 248,
            'endLine' => 248,
            'startColumn' => 37,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'max' => 
          array (
            'name' => 'max',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 248,
            'endLine' => 248,
            'startColumn' => 49,
            'endColumn' => 58,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 248,
            'endLine' => 248,
            'startColumn' => 61,
            'endColumn' => 66,
            'parameterIndex' => 2,
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
 * Return a vector of n evenly spaced numbers between minimum and maximum.
 *
 * @param float $min
 * @param float $max
 * @param int $n
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 248,
        'endLine' => 273,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'a' => 
          array (
            'name' => 'a',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 279,
            'endLine' => 279,
            'startColumn' => 39,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'validate' => 
          array (
            'name' => 'validate',
            'default' => 
            array (
              'code' => 'true',
              'attributes' => 
              array (
                'startLine' => 279,
                'endLine' => 279,
                'startTokenPos' => 1197,
                'startFilePos' => 6200,
                'endTokenPos' => 1197,
                'endFilePos' => 6203,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'bool',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 279,
            'endLine' => 279,
            'startColumn' => 49,
            'endColumn' => 69,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param mixed[] $a
 * @param bool $validate
 */',
        'startLine' => 279,
        'endLine' => 293,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 33,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'shape' => 
      array (
        'name' => 'shape',
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
 * Return a tuple with the dimensionality of the tensor.
 *
 * @return array{int<0,max>}
 */',
        'startLine' => 300,
        'endLine' => 303,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'shapeString' => 
      array (
        'name' => 'shapeString',
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
 * Return the shape of the tensor as a string.
 *
 * @return string
 */',
        'startLine' => 310,
        'endLine' => 313,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'size' => 
      array (
        'name' => 'size',
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
 * Return the number of elements in the vector.
 *
 * @return int<0,max>
 */',
        'startLine' => 320,
        'endLine' => 323,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'm' => 
      array (
        'name' => 'm',
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
 * Return the number of rows in the vector.
 *
 * @return int<0,max>
 */',
        'startLine' => 330,
        'endLine' => 333,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'n' => 
      array (
        'name' => 'n',
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
 * Return the number of columns in the vector.
 *
 * @return int<0,max>
 */',
        'startLine' => 340,
        'endLine' => 343,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'asArray' => 
      array (
        'name' => 'asArray',
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
 * Return the vector as an array.
 *
 * @return list<float>
 */',
        'startLine' => 350,
        'endLine' => 353,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'asRowMatrix' => 
      array (
        'name' => 'asRowMatrix',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return this vector as a row matrix.
 *
 * @return Matrix
 */',
        'startLine' => 360,
        'endLine' => 363,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'asColumnMatrix' => 
      array (
        'name' => 'asColumnMatrix',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return this vector as a column matrix.
 *
 * @return Matrix
 */',
        'startLine' => 370,
        'endLine' => 379,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'reshape' => 
      array (
        'name' => 'reshape',
        'parameters' => 
        array (
          'm' => 
          array (
            'name' => 'm',
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
            'startLine' => 389,
            'endLine' => 389,
            'startColumn' => 29,
            'endColumn' => 34,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'n' => 
          array (
            'name' => 'n',
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
            'startLine' => 389,
            'endLine' => 389,
            'startColumn' => 37,
            'endColumn' => 42,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return a matrix in the shape specified.
 *
 * @param int $m
 * @param int $n
 * @throws InvalidArgumentException
 * @return Matrix
 */',
        'startLine' => 389,
        'endLine' => 413,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'transpose' => 
      array (
        'name' => 'transpose',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Transpose the vector i.e. rotate it.
 *
 * @return mixed
 */',
        'startLine' => 420,
        'endLine' => 423,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'map' => 
      array (
        'name' => 'map',
        'parameters' => 
        array (
          'callback' => 
          array (
            'name' => 'callback',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'callable',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 433,
            'endLine' => 433,
            'startColumn' => 25,
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
 * Map a function over the elements in the vector and return a new vector.
 *
 * @internal
 *
 * @param callable $callback
 * @return self
 */',
        'startLine' => 433,
        'endLine' => 436,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'reduce' => 
      array (
        'name' => 'reduce',
        'parameters' => 
        array (
          'callback' => 
          array (
            'name' => 'callback',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'callable',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 447,
            'endLine' => 447,
            'startColumn' => 28,
            'endColumn' => 45,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'initial' => 
          array (
            'name' => 'initial',
            'default' => 
            array (
              'code' => '0.0',
              'attributes' => 
              array (
                'startLine' => 447,
                'endLine' => 447,
                'startTokenPos' => 1787,
                'startFilePos' => 9432,
                'endTokenPos' => 1787,
                'endFilePos' => 9434,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 447,
            'endLine' => 447,
            'startColumn' => 48,
            'endColumn' => 67,
            'parameterIndex' => 1,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Reduce the vector down to a scalar.
 *
 * @internal
 *
 * @param callable $callback
 * @param float $initial
 * @return float
 */',
        'startLine' => 447,
        'endLine' => 450,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'dot' => 
      array (
        'name' => 'dot',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 459,
            'endLine' => 459,
            'startColumn' => 25,
            'endColumn' => 33,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the dot product of this vector and another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return float
 */',
        'startLine' => 459,
        'endLine' => 473,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'matmul' => 
      array (
        'name' => 'matmul',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 481,
            'endLine' => 481,
            'startColumn' => 28,
            'endColumn' => 36,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Compute the vector-matrix product of this vector and matrix a.
 *
 * @param Matrix $b
 * @return Matrix
 */',
        'startLine' => 481,
        'endLine' => 484,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'inner' => 
      array (
        'name' => 'inner',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 492,
            'endLine' => 492,
            'startColumn' => 27,
            'endColumn' => 35,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the inner product of two vectors.
 *
 * @param Vector $b
 * @return float
 */',
        'startLine' => 492,
        'endLine' => 495,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'outer' => 
      array (
        'name' => 'outer',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 503,
            'endLine' => 503,
            'startColumn' => 27,
            'endColumn' => 35,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the outer product of this and another vector.
 *
 * @param Vector $b
 * @return Matrix
 */',
        'startLine' => 503,
        'endLine' => 520,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'convolve' => 
      array (
        'name' => 'convolve',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 530,
            'endLine' => 530,
            'startColumn' => 30,
            'endColumn' => 38,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'stride' => 
          array (
            'name' => 'stride',
            'default' => 
            array (
              'code' => '1',
              'attributes' => 
              array (
                'startLine' => 530,
                'endLine' => 530,
                'startTokenPos' => 2126,
                'startFilePos' => 11282,
                'endTokenPos' => 2126,
                'endFilePos' => 11282,
              ),
            ),
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
            'startLine' => 530,
            'endLine' => 530,
            'startColumn' => 41,
            'endColumn' => 55,
            'parameterIndex' => 1,
            'isOptional' => true,
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
 * Return the 1D convolution of this vector and a kernel vector with given stride.
 *
 * @param Vector $b
 * @param int $stride
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 530,
        'endLine' => 564,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'l1Norm' => 
      array (
        'name' => 'l1Norm',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the L1 or Manhattan norm of the vector.
 *
 * @return float
 */',
        'startLine' => 571,
        'endLine' => 574,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'l2Norm' => 
      array (
        'name' => 'l2Norm',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the L2 or Euclidean norm of the vector.
 *
 * @return float
 */',
        'startLine' => 581,
        'endLine' => 584,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'pNorm' => 
      array (
        'name' => 'pNorm',
        'parameters' => 
        array (
          'p' => 
          array (
            'name' => 'p',
            'default' => 
            array (
              'code' => '3.0',
              'attributes' => 
              array (
                'startLine' => 593,
                'endLine' => 593,
                'startTokenPos' => 2487,
                'startFilePos' => 12670,
                'endTokenPos' => 2487,
                'endFilePos' => 12672,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 593,
            'endLine' => 593,
            'startColumn' => 27,
            'endColumn' => 40,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the p-norm of the vector.
 *
 * @param float $p
 * @throws InvalidArgumentException
 * @return float
 */',
        'startLine' => 593,
        'endLine' => 600,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'maxNorm' => 
      array (
        'name' => 'maxNorm',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Calculate the max norm of the vector.
 *
 * @return float
 */',
        'startLine' => 607,
        'endLine' => 610,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'multiply' => 
      array (
        'name' => 'multiply',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 619,
            'endLine' => 619,
            'startColumn' => 30,
            'endColumn' => 31,
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
 * A universal function to multiply this vector with another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 619,
        'endLine' => 640,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'divide' => 
      array (
        'name' => 'divide',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 649,
            'endLine' => 649,
            'startColumn' => 28,
            'endColumn' => 29,
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
 * A universal function to divide this vector by another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 649,
        'endLine' => 670,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'add' => 
      array (
        'name' => 'add',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 679,
            'endLine' => 679,
            'startColumn' => 25,
            'endColumn' => 26,
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
 * A universal function to add this vector with another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 679,
        'endLine' => 700,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'subtract' => 
      array (
        'name' => 'subtract',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 709,
            'endLine' => 709,
            'startColumn' => 30,
            'endColumn' => 31,
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
 * A universal function to subtract this vector from another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 709,
        'endLine' => 730,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'pow' => 
      array (
        'name' => 'pow',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 740,
            'endLine' => 740,
            'startColumn' => 25,
            'endColumn' => 26,
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
 * A universal function to raise this vector to the power of another
 * tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 740,
        'endLine' => 761,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'mod' => 
      array (
        'name' => 'mod',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 771,
            'endLine' => 771,
            'startColumn' => 25,
            'endColumn' => 26,
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
 * A universal function to compute the modulus of this vector and
 * another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 771,
        'endLine' => 792,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'equal' => 
      array (
        'name' => 'equal',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 802,
            'endLine' => 802,
            'startColumn' => 27,
            'endColumn' => 28,
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
 * A universal function to compute the equality comparison of
 * this vector and another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 802,
        'endLine' => 823,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'notEqual' => 
      array (
        'name' => 'notEqual',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 833,
            'endLine' => 833,
            'startColumn' => 30,
            'endColumn' => 31,
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
 * A universal function to compute the not equal comparison of
 * this vector and another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 833,
        'endLine' => 854,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greater' => 
      array (
        'name' => 'greater',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 864,
            'endLine' => 864,
            'startColumn' => 29,
            'endColumn' => 30,
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
 * A universal function to compute the greater than comparison of
 * this vector and another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 864,
        'endLine' => 885,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterEqual' => 
      array (
        'name' => 'greaterEqual',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 895,
            'endLine' => 895,
            'startColumn' => 34,
            'endColumn' => 35,
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
 * A universal function to compute the greater than or equal to
 * comparison of this vector and another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 895,
        'endLine' => 916,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'less' => 
      array (
        'name' => 'less',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 926,
            'endLine' => 926,
            'startColumn' => 26,
            'endColumn' => 27,
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
 * A universal function to compute the less than comparison of
 * this vector and another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 926,
        'endLine' => 947,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessEqual' => 
      array (
        'name' => 'lessEqual',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 957,
            'endLine' => 957,
            'startColumn' => 31,
            'endColumn' => 32,
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
 * A universal function to compute the less than or equal to
 * comparison of this vector and another tensor element-wise.
 *
 * @param mixed $b
 * @throws InvalidArgumentException
 * @return mixed
 */',
        'startLine' => 957,
        'endLine' => 978,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'reciprocal' => 
      array (
        'name' => 'reciprocal',
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
 * Return the reciprocal of the vector element-wise.
 *
 * @return self
 */',
        'startLine' => 985,
        'endLine' => 989,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'abs' => 
      array (
        'name' => 'abs',
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
 * Return the absolute value of the vector.
 *
 * @return self
 */',
        'startLine' => 996,
        'endLine' => 999,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'square' => 
      array (
        'name' => 'square',
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
 * Square the vector.
 *
 * @return self
 */',
        'startLine' => 1006,
        'endLine' => 1009,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'sqrt' => 
      array (
        'name' => 'sqrt',
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
 * Take the square root of the vector.
 *
 * @return self
 */',
        'startLine' => 1016,
        'endLine' => 1019,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'exp' => 
      array (
        'name' => 'exp',
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
 * Exponentiate each element in the vector.
 *
 * @return self
 */',
        'startLine' => 1026,
        'endLine' => 1029,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'expm1' => 
      array (
        'name' => 'expm1',
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
 * Return the exponential of the tensor minus 1.
 *
 * @return self
 */',
        'startLine' => 1036,
        'endLine' => 1039,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'log' => 
      array (
        'name' => 'log',
        'parameters' => 
        array (
          'base' => 
          array (
            'name' => 'base',
            'default' => 
            array (
              'code' => 'M_E',
              'attributes' => 
              array (
                'startLine' => 1047,
                'endLine' => 1047,
                'startTokenPos' => 4193,
                'startFilePos' => 24176,
                'endTokenPos' => 4193,
                'endFilePos' => 24178,
              ),
            ),
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1047,
            'endLine' => 1047,
            'startColumn' => 25,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => true,
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
 * Take the log to the given base of each element in the vector.
 *
 * @param float $base
 * @return self
 */',
        'startLine' => 1047,
        'endLine' => 1056,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'log1p' => 
      array (
        'name' => 'log1p',
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
 * Return the log of 1 plus the tensor.
 *
 * @return self
 */',
        'startLine' => 1063,
        'endLine' => 1066,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'sin' => 
      array (
        'name' => 'sin',
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
 * Return the sine of this vector.
 *
 * @return self
 */',
        'startLine' => 1073,
        'endLine' => 1076,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'asin' => 
      array (
        'name' => 'asin',
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
 * Compute the arc sine of the vector.
 *
 * @return self
 */',
        'startLine' => 1083,
        'endLine' => 1086,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'cos' => 
      array (
        'name' => 'cos',
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
 * Return the cosine of this vector.
 *
 * @return self
 */',
        'startLine' => 1093,
        'endLine' => 1096,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'acos' => 
      array (
        'name' => 'acos',
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
 * Compute the arc cosine of the vector.
 *
 * @return self
 */',
        'startLine' => 1103,
        'endLine' => 1106,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'tan' => 
      array (
        'name' => 'tan',
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
 * Return the tangent of this vector.
 *
 * @return self
 */',
        'startLine' => 1113,
        'endLine' => 1116,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'atan' => 
      array (
        'name' => 'atan',
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
 * Compute the arc tangent of the vector.
 *
 * @return self
 */',
        'startLine' => 1123,
        'endLine' => 1126,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'rad2deg' => 
      array (
        'name' => 'rad2deg',
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
 * Convert angles from radians to degrees.
 *
 * @return self
 */',
        'startLine' => 1133,
        'endLine' => 1136,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'deg2rad' => 
      array (
        'name' => 'deg2rad',
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
 * Convert angles from degrees to radians.
 *
 * @return self
 */',
        'startLine' => 1143,
        'endLine' => 1146,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'sum' => 
      array (
        'name' => 'sum',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The sum of the vector.
 *
 * @return float
 */',
        'startLine' => 1153,
        'endLine' => 1156,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'product' => 
      array (
        'name' => 'product',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the product of the vector.
 *
 * @return float
 */',
        'startLine' => 1163,
        'endLine' => 1166,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'min' => 
      array (
        'name' => 'min',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the minimum element in the vector.
 *
 * @return float
 */',
        'startLine' => 1173,
        'endLine' => 1176,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'max' => 
      array (
        'name' => 'max',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the maximum element in the vector.
 *
 * @return float
 */',
        'startLine' => 1183,
        'endLine' => 1186,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'mean' => 
      array (
        'name' => 'mean',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the mean of the vector.
 *
 * @return float
 */',
        'startLine' => 1193,
        'endLine' => 1196,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'median' => 
      array (
        'name' => 'median',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the median of the vector.
 *
 * @return float
 */',
        'startLine' => 1203,
        'endLine' => 1218,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'quantile' => 
      array (
        'name' => 'quantile',
        'parameters' => 
        array (
          'q' => 
          array (
            'name' => 'q',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1227,
            'endLine' => 1227,
            'startColumn' => 30,
            'endColumn' => 37,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the q\'th quantile of the vector.
 *
 * @param float $q
 * @throws InvalidArgumentException
 * @return float
 */',
        'startLine' => 1227,
        'endLine' => 1247,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'variance' => 
      array (
        'name' => 'variance',
        'parameters' => 
        array (
          'mean' => 
          array (
            'name' => 'mean',
            'default' => 
            array (
              'code' => 'null',
              'attributes' => 
              array (
                'startLine' => 1255,
                'endLine' => 1255,
                'startTokenPos' => 4937,
                'startFilePos' => 27865,
                'endTokenPos' => 4937,
                'endFilePos' => 27868,
              ),
            ),
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1255,
            'endLine' => 1255,
            'startColumn' => 30,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => true,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the variance of the vector.
 *
 * @param float|null $mean
 * @return float
 */',
        'startLine' => 1255,
        'endLine' => 1266,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'round' => 
      array (
        'name' => 'round',
        'parameters' => 
        array (
          'precision' => 
          array (
            'name' => 'precision',
            'default' => 
            array (
              'code' => '0',
              'attributes' => 
              array (
                'startLine' => 1275,
                'endLine' => 1275,
                'startTokenPos' => 5020,
                'startFilePos' => 28310,
                'endTokenPos' => 5020,
                'endFilePos' => 28310,
              ),
            ),
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
            'startLine' => 1275,
            'endLine' => 1275,
            'startColumn' => 27,
            'endColumn' => 44,
            'parameterIndex' => 0,
            'isOptional' => true,
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
 * Round the elements in the matrix to a given decimal place.
 *
 * @param int $precision
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 1275,
        'endLine' => 1289,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'floor' => 
      array (
        'name' => 'floor',
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
 * Round the elements in the vector down to the nearest integer.
 *
 * @return self
 */',
        'startLine' => 1296,
        'endLine' => 1299,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'ceil' => 
      array (
        'name' => 'ceil',
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
 * Round the elements in the vector up to the nearest integer.
 *
 * @return self
 */',
        'startLine' => 1306,
        'endLine' => 1309,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'clip' => 
      array (
        'name' => 'clip',
        'parameters' => 
        array (
          'min' => 
          array (
            'name' => 'min',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1320,
            'endLine' => 1320,
            'startColumn' => 26,
            'endColumn' => 35,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'max' => 
          array (
            'name' => 'max',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1320,
            'endLine' => 1320,
            'startColumn' => 38,
            'endColumn' => 47,
            'parameterIndex' => 1,
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
 * Clip the elements in the vector to be between given minimum and maximum
 * and return a new vector.
 *
 * @param float $min
 * @param float $max
 * @throws InvalidArgumentException
 * @return self
 */',
        'startLine' => 1320,
        'endLine' => 1346,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'clipLower' => 
      array (
        'name' => 'clipLower',
        'parameters' => 
        array (
          'min' => 
          array (
            'name' => 'min',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1354,
            'endLine' => 1354,
            'startColumn' => 31,
            'endColumn' => 40,
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
 * Clip the tensor to be lower bounded by a given minimum.
 *
 * @param float $min
 * @return self
 */',
        'startLine' => 1354,
        'endLine' => 1369,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'clipUpper' => 
      array (
        'name' => 'clipUpper',
        'parameters' => 
        array (
          'max' => 
          array (
            'name' => 'max',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1377,
            'endLine' => 1377,
            'startColumn' => 31,
            'endColumn' => 40,
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
 * Clip the tensor to be upper bounded by a given maximum.
 *
 * @param float $max
 * @return self
 */',
        'startLine' => 1377,
        'endLine' => 1392,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'sign' => 
      array (
        'name' => 'sign',
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
 * Return the element-wise sign indication.
 *
 * @return self
 */',
        'startLine' => 1399,
        'endLine' => 1414,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'negate' => 
      array (
        'name' => 'negate',
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
 * Negate the vector i.e take the negative of each value element-wise.
 *
 * @return self
 */',
        'startLine' => 1421,
        'endLine' => 1430,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'multiplyMatrix' => 
      array (
        'name' => 'multiplyMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1439,
            'endLine' => 1439,
            'startColumn' => 36,
            'endColumn' => 44,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Multiply this vector with a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1439,
        'endLine' => 1459,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'divideMatrix' => 
      array (
        'name' => 'divideMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1468,
            'endLine' => 1468,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Divide this vector with a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1468,
        'endLine' => 1488,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'addMatrix' => 
      array (
        'name' => 'addMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1497,
            'endLine' => 1497,
            'startColumn' => 31,
            'endColumn' => 39,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Add this vector to a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1497,
        'endLine' => 1517,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'subtractMatrix' => 
      array (
        'name' => 'subtractMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1526,
            'endLine' => 1526,
            'startColumn' => 36,
            'endColumn' => 44,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Subtract a matrix from this vector.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1526,
        'endLine' => 1546,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'powMatrix' => 
      array (
        'name' => 'powMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1555,
            'endLine' => 1555,
            'startColumn' => 31,
            'endColumn' => 39,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Raise this vector to the power of a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1555,
        'endLine' => 1575,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'modMatrix' => 
      array (
        'name' => 'modMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1584,
            'endLine' => 1584,
            'startColumn' => 31,
            'endColumn' => 39,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Mod this vector with a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1584,
        'endLine' => 1604,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'equalMatrix' => 
      array (
        'name' => 'equalMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1613,
            'endLine' => 1613,
            'startColumn' => 33,
            'endColumn' => 41,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the element-wise equality comparison of this vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1613,
        'endLine' => 1633,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'notEqualMatrix' => 
      array (
        'name' => 'notEqualMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1642,
            'endLine' => 1642,
            'startColumn' => 36,
            'endColumn' => 44,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the element-wise not equal comparison of this vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1642,
        'endLine' => 1662,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterMatrix' => 
      array (
        'name' => 'greaterMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1671,
            'endLine' => 1671,
            'startColumn' => 35,
            'endColumn' => 43,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the element-wise greater than comparison of this vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1671,
        'endLine' => 1691,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterEqualMatrix' => 
      array (
        'name' => 'greaterEqualMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1700,
            'endLine' => 1700,
            'startColumn' => 40,
            'endColumn' => 48,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the element-wise greater than or equal to comparison of this vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1700,
        'endLine' => 1720,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessMatrix' => 
      array (
        'name' => 'lessMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1729,
            'endLine' => 1729,
            'startColumn' => 32,
            'endColumn' => 40,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the element-wise less than comparison of this vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1729,
        'endLine' => 1749,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessEqualMatrix' => 
      array (
        'name' => 'lessEqualMatrix',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Matrix',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1758,
            'endLine' => 1758,
            'startColumn' => 37,
            'endColumn' => 45,
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
            'name' => 'Tensor\\Matrix',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the element-wise less than or equal to comparison of this vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 1758,
        'endLine' => 1778,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'multiplyVector' => 
      array (
        'name' => 'multiplyVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1787,
            'endLine' => 1787,
            'startColumn' => 36,
            'endColumn' => 44,
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
 * Multiply this vector with another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1787,
        'endLine' => 1801,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'divideVector' => 
      array (
        'name' => 'divideVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1810,
            'endLine' => 1810,
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
 * Divide this vector by another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1810,
        'endLine' => 1824,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'addVector' => 
      array (
        'name' => 'addVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1833,
            'endLine' => 1833,
            'startColumn' => 31,
            'endColumn' => 39,
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
 * Add this vector to another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1833,
        'endLine' => 1847,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'subtractVector' => 
      array (
        'name' => 'subtractVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1856,
            'endLine' => 1856,
            'startColumn' => 36,
            'endColumn' => 44,
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
 * Subtract a vector from this vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1856,
        'endLine' => 1870,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'powVector' => 
      array (
        'name' => 'powVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1879,
            'endLine' => 1879,
            'startColumn' => 31,
            'endColumn' => 39,
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
 * Raise this vector to a power of another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1879,
        'endLine' => 1893,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'modVector' => 
      array (
        'name' => 'modVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1902,
            'endLine' => 1902,
            'startColumn' => 31,
            'endColumn' => 39,
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
 * Calculate the modulus of this vector with another vector element-wise.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1902,
        'endLine' => 1916,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'equalVector' => 
      array (
        'name' => 'equalVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1925,
            'endLine' => 1925,
            'startColumn' => 33,
            'endColumn' => 41,
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
 * Return the element-wise equality comparison of this vector and a another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1925,
        'endLine' => 1939,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'notEqualVector' => 
      array (
        'name' => 'notEqualVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1948,
            'endLine' => 1948,
            'startColumn' => 36,
            'endColumn' => 44,
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
 * Return the element-wise not equal comparison of this vector and a another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1948,
        'endLine' => 1962,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterVector' => 
      array (
        'name' => 'greaterVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1971,
            'endLine' => 1971,
            'startColumn' => 35,
            'endColumn' => 43,
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
 * Return the element-wise greater than comparison of this vector and a another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return self
 */',
        'startLine' => 1971,
        'endLine' => 1985,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterEqualVector' => 
      array (
        'name' => 'greaterEqualVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 1994,
            'endLine' => 1994,
            'startColumn' => 40,
            'endColumn' => 48,
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
 * Return the element-wise greater than or equal to comparison of this vector and a another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 1994,
        'endLine' => 2008,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessVector' => 
      array (
        'name' => 'lessVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2017,
            'endLine' => 2017,
            'startColumn' => 32,
            'endColumn' => 40,
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
 * Return the element-wise less than comparison of this vector and a another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 2017,
        'endLine' => 2031,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessEqualVector' => 
      array (
        'name' => 'lessEqualVector',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Tensor\\Vector',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2040,
            'endLine' => 2040,
            'startColumn' => 37,
            'endColumn' => 45,
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
 * Return the element-wise less than or equal to comparison of this vector and a another vector.
 *
 * @param Vector $b
 * @throws DimensionalityMismatch
 * @return static
 */',
        'startLine' => 2040,
        'endLine' => 2054,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'multiplyScalar' => 
      array (
        'name' => 'multiplyScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2062,
            'endLine' => 2062,
            'startColumn' => 36,
            'endColumn' => 43,
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
 * Multiply this vector by a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2062,
        'endLine' => 2071,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'divideScalar' => 
      array (
        'name' => 'divideScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2079,
            'endLine' => 2079,
            'startColumn' => 34,
            'endColumn' => 41,
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
 * Divide this vector by a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2079,
        'endLine' => 2088,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'addScalar' => 
      array (
        'name' => 'addScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2096,
            'endLine' => 2096,
            'startColumn' => 31,
            'endColumn' => 38,
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
 * Add a scalar to this vector.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2096,
        'endLine' => 2105,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'subtractScalar' => 
      array (
        'name' => 'subtractScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2113,
            'endLine' => 2113,
            'startColumn' => 36,
            'endColumn' => 43,
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
 * Subtract a scalar from this vector.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2113,
        'endLine' => 2122,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'powScalar' => 
      array (
        'name' => 'powScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2130,
            'endLine' => 2130,
            'startColumn' => 31,
            'endColumn' => 38,
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
 * Raise the vector to a the power of a scalar value.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2130,
        'endLine' => 2139,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'modScalar' => 
      array (
        'name' => 'modScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2147,
            'endLine' => 2147,
            'startColumn' => 31,
            'endColumn' => 38,
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
 * Calculate the modulus of this vector with a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2147,
        'endLine' => 2156,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'equalScalar' => 
      array (
        'name' => 'equalScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2164,
            'endLine' => 2164,
            'startColumn' => 33,
            'endColumn' => 40,
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
 * Return the element-wise equality comparison of this vector and a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2164,
        'endLine' => 2173,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'notEqualScalar' => 
      array (
        'name' => 'notEqualScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2181,
            'endLine' => 2181,
            'startColumn' => 36,
            'endColumn' => 43,
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
 * Return the element-wise not equal comparison of this vector and a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2181,
        'endLine' => 2190,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterScalar' => 
      array (
        'name' => 'greaterScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2198,
            'endLine' => 2198,
            'startColumn' => 35,
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
 * Return the element-wise greater than comparison of this vector and a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2198,
        'endLine' => 2207,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'greaterEqualScalar' => 
      array (
        'name' => 'greaterEqualScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2215,
            'endLine' => 2215,
            'startColumn' => 40,
            'endColumn' => 47,
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
 * Return the element-wise greater than or equal to comparison of this vector and a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2215,
        'endLine' => 2224,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessScalar' => 
      array (
        'name' => 'lessScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2232,
            'endLine' => 2232,
            'startColumn' => 32,
            'endColumn' => 39,
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
 * Return the element-wise less than comparison of this vector and a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2232,
        'endLine' => 2241,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'lessEqualScalar' => 
      array (
        'name' => 'lessEqualScalar',
        'parameters' => 
        array (
          'b' => 
          array (
            'name' => 'b',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'float',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2249,
            'endLine' => 2249,
            'startColumn' => 37,
            'endColumn' => 44,
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
 * Return the element-wise less than or equal to comparison of this vector and a scalar.
 *
 * @param float $b
 * @return static
 */',
        'startLine' => 2249,
        'endLine' => 2258,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'count' => 
      array (
        'name' => 'count',
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
 * Count method to implement countable interface.
 *
 * @return int
 */',
        'startLine' => 2265,
        'endLine' => 2268,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'offsetSet' => 
      array (
        'name' => 'offsetSet',
        'parameters' => 
        array (
          'index' => 
          array (
            'name' => 'index',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2276,
            'endLine' => 2276,
            'startColumn' => 31,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'values' => 
          array (
            'name' => 'values',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2276,
            'endLine' => 2276,
            'startColumn' => 39,
            'endColumn' => 45,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'ReturnTypeWillChange',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * @param mixed $index
 * @param mixed[] $values
 * @throws RuntimeException
 */',
        'startLine' => 2275,
        'endLine' => 2279,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'offsetExists' => 
      array (
        'name' => 'offsetExists',
        'parameters' => 
        array (
          'index' => 
          array (
            'name' => 'index',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2287,
            'endLine' => 2287,
            'startColumn' => 34,
            'endColumn' => 39,
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
            'name' => 'bool',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Does a given column exist in the matrix.
 *
 * @param mixed $index
 * @return bool
 */',
        'startLine' => 2287,
        'endLine' => 2290,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'offsetUnset' => 
      array (
        'name' => 'offsetUnset',
        'parameters' => 
        array (
          'index' => 
          array (
            'name' => 'index',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2296,
            'endLine' => 2296,
            'startColumn' => 33,
            'endColumn' => 38,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param mixed $index
 * @throws RuntimeException
 */',
        'startLine' => 2296,
        'endLine' => 2299,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'offsetGet' => 
      array (
        'name' => 'offsetGet',
        'parameters' => 
        array (
          'index' => 
          array (
            'name' => 'index',
            'default' => NULL,
            'type' => NULL,
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 2309,
            'endLine' => 2309,
            'startColumn' => 31,
            'endColumn' => 36,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'ReturnTypeWillChange',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Return a row from the matrix at the given index.
 *
 * @param mixed $index
 * @throws InvalidArgumentException
 * @return float
 */',
        'startLine' => 2308,
        'endLine' => 2317,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
        'aliasName' => NULL,
      ),
      'getIterator' => 
      array (
        'name' => 'getIterator',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Traversable',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
          0 => 
          array (
            'name' => 'ReturnTypeWillChange',
            'isRepeated' => false,
            'arguments' => 
            array (
            ),
          ),
        ),
        'docComment' => '/**
 * Get an iterator for the rows in the matrix.
 *
 * @return \\ArrayIterator<int,float>
 */',
        'startLine' => 2324,
        'endLine' => 2328,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Vector',
        'implementingClassName' => 'Tensor\\Vector',
        'currentClassName' => 'Tensor\\Vector',
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