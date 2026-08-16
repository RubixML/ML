<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/functions.php-PHPStan\BetterReflection\Reflection\ReflectionFunction-Rubix\ML\iterator_contains_nan
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-1a4a61157c47de52bea1079cacf2ded805a853d420f5790f53125682c8ffedcf',
   'data' => 
  array (
    'name' => 'iterator_contains_nan',
    'parameters' => 
    array (
      'values' => 
      array (
        'name' => 'values',
        'default' => NULL,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'iterable',
            'isIdentifier' => true,
          ),
        ),
        'isVariadic' => false,
        'byRef' => false,
        'isPromoted' => false,
        'attributes' => 
        array (
        ),
        'startLine' => 220,
        'endLine' => 220,
        'startColumn' => 36,
        'endColumn' => 51,
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
 * Check if an iterator contains NAN values recursively.
 *
 * @internal
 *
 * @param iterable<mixed> $values
 * @return bool
 */',
    'startLine' => 220,
    'endLine' => 237,
    'startColumn' => 5,
    'endColumn' => 5,
    'couldThrow' => false,
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
        'name' => 'Rubix\\ML\\iterator_contains_nan',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/functions.php',
      ),
    ),
  ),
));