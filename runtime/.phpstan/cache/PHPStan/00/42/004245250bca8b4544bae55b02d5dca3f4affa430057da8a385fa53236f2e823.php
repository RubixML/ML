<?php declare(strict_types = 1);

// osfsl-/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/ColumnVector.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Tensor\ColumnVector
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-184b7450afa59079d17986a3d6da791868f73fd9eb1a36b6b87c5b6c412f5817-8.4-6.70.0.3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Tensor\\ColumnVector',
        'filename' => '/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/ColumnVector.php',
      ),
    ),
    'namespace' => 'Tensor',
    'name' => 'Tensor\\ColumnVector',
    'shortName' => 'ColumnVector',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Column Vector
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
    'startLine' => 16,
    'endLine' => 436,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => 'Tensor\\Vector',
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
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
 * @return int
 */',
        'startLine' => 23,
        'endLine' => 26,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
 * @return int
 */',
        'startLine' => 33,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
        'startLine' => 43,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 54,
            'endLine' => 54,
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
        'startLine' => 54,
        'endLine' => 57,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 66,
            'endLine' => 66,
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
 * Multiply this column vector with a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 66,
        'endLine' => 88,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 97,
            'endLine' => 97,
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
 * Divide this column vector with a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 97,
        'endLine' => 119,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 128,
            'endLine' => 128,
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
 * Add this column vector to a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 128,
        'endLine' => 150,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 159,
            'endLine' => 159,
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
 * Subtract a matrix from this column vector.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 159,
        'endLine' => 181,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 190,
            'endLine' => 190,
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
 * Raise this column vector to the power of a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 190,
        'endLine' => 212,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 221,
            'endLine' => 221,
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
 * Mod this column vector with a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 221,
        'endLine' => 243,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 253,
            'endLine' => 253,
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
 * Return the element-wise equality comparison of this column vector
 * and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 253,
        'endLine' => 275,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 285,
            'endLine' => 285,
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
 * Return the element-wise not equal comparison of this column vector
 * and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 285,
        'endLine' => 307,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 317,
            'endLine' => 317,
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
 * Return the element-wise greater than comparison of this column
 * vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 317,
        'endLine' => 339,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 349,
            'endLine' => 349,
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
 * Return the element-wise greater than or equal to comparison of
 * this column vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 349,
        'endLine' => 371,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 381,
            'endLine' => 381,
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
 * Return the element-wise less than comparison of this column
 * vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 381,
        'endLine' => 403,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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
            'startLine' => 413,
            'endLine' => 413,
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
 * Return the element-wise less than or equal to comparison of
 * this column vector and a matrix.
 *
 * @param Matrix $b
 * @throws DimensionalityMismatch
 * @return Matrix
 */',
        'startLine' => 413,
        'endLine' => 435,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\ColumnVector',
        'implementingClassName' => 'Tensor\\ColumnVector',
        'currentClassName' => 'Tensor\\ColumnVector',
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