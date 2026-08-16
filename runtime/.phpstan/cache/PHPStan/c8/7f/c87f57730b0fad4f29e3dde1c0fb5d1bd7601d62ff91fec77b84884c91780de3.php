<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/functions.php-PHPStan\BetterReflection\Reflection\ReflectionFunction-Rubix\ML\linspace
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-1a4a61157c47de52bea1079cacf2ded805a853d420f5790f53125682c8ffedcf',
   'data' => 
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
        'startLine' => 111,
        'endLine' => 111,
        'startColumn' => 23,
        'endColumn' => 32,
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
        'startLine' => 111,
        'endLine' => 111,
        'startColumn' => 35,
        'endColumn' => 44,
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
        'startLine' => 111,
        'endLine' => 111,
        'startColumn' => 47,
        'endColumn' => 52,
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
        'name' => 'array',
        'isIdentifier' => true,
      ),
    ),
    'attributes' => 
    array (
    ),
    'docComment' => '/**
 * Return an array of n evenly spaced numbers between minimum and maximum.
 *
 * @param float $min
 * @param float $max
 * @param int $n
 * @throws \\Tensor\\Exceptions\\InvalidArgumentException
 * @return list<float>
 */',
    'startLine' => 111,
    'endLine' => 136,
    'startColumn' => 5,
    'endColumn' => 5,
    'couldThrow' => true,
    'isClosure' => false,
    'isGenerator' => false,
    'isVariadic' => false,
    'isStatic' => false,
    'namespace' => 'Rubix\\ML',
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\linspace',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/functions.php',
      ),
    ),
  ),
));