<?php declare(strict_types = 1);

// osfsl-/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Arithmetic.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Tensor\Arithmetic
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-d32e36e8fdd2f5d9f23207f78cd5ea490e376a0f5ba0cbeec9ba7ce4f273ac6f-8.4-6.70.0.3',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Tensor\\Arithmetic',
        'filename' => '/home/andrew/Workspace/Rubix/ML/vendor/composer/../rubix/tensor/src/Arithmetic.php',
      ),
    ),
    'namespace' => 'Tensor',
    'name' => 'Tensor\\Arithmetic',
    'shortName' => 'Arithmetic',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => NULL,
    'attributes' => 
    array (
    ),
    'startLine' => 5,
    'endLine' => 54,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
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
            'startLine' => 13,
            'endLine' => 13,
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
 * A universal function to multiply this tensor with another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 13,
        'endLine' => 13,
        'startColumn' => 5,
        'endColumn' => 33,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Arithmetic',
        'implementingClassName' => 'Tensor\\Arithmetic',
        'currentClassName' => 'Tensor\\Arithmetic',
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
            'startLine' => 21,
            'endLine' => 21,
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
 * A universal function to divide this tensor by another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 21,
        'endLine' => 21,
        'startColumn' => 5,
        'endColumn' => 31,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Arithmetic',
        'implementingClassName' => 'Tensor\\Arithmetic',
        'currentClassName' => 'Tensor\\Arithmetic',
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
            'startLine' => 29,
            'endLine' => 29,
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
 * A universal function to add this tensor with another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 29,
        'endLine' => 29,
        'startColumn' => 5,
        'endColumn' => 28,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Arithmetic',
        'implementingClassName' => 'Tensor\\Arithmetic',
        'currentClassName' => 'Tensor\\Arithmetic',
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
            'startLine' => 37,
            'endLine' => 37,
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
 * A universal function to subtract this tensor from another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 37,
        'endLine' => 37,
        'startColumn' => 5,
        'endColumn' => 33,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Arithmetic',
        'implementingClassName' => 'Tensor\\Arithmetic',
        'currentClassName' => 'Tensor\\Arithmetic',
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
            'startLine' => 45,
            'endLine' => 45,
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
 * A universal function to raise this tensor to the power of another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 45,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 28,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Arithmetic',
        'implementingClassName' => 'Tensor\\Arithmetic',
        'currentClassName' => 'Tensor\\Arithmetic',
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
            'startLine' => 53,
            'endLine' => 53,
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
 * A universal function to compute the integer modulus of this tensor and another tensor element-wise.
 *
 * @param mixed $b
 * @return mixed
 */',
        'startLine' => 53,
        'endLine' => 53,
        'startColumn' => 5,
        'endColumn' => 28,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Tensor',
        'declaringClassName' => 'Tensor\\Arithmetic',
        'implementingClassName' => 'Tensor\\Arithmetic',
        'currentClassName' => 'Tensor\\Arithmetic',
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