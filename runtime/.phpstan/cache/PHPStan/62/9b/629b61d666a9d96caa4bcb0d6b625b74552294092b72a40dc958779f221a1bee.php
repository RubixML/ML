<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Strategies/KMostFrequent.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Strategies\KMostFrequent
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-81b0e2c81e36681d00590e91ee62adfbdbccba34005e0a4e77ee4961fdc714bc',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Strategies/KMostFrequent.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Strategies',
    'name' => 'Rubix\\ML\\Strategies\\KMostFrequent',
    'shortName' => 'KMostFrequent',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * K Most Frequent
 *
 * This Strategy outputs one of k most frequently occurring classes at random with equal probability.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 20,
    'endLine' => 128,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Strategies\\Strategy',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'k' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'name' => 'k',
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
 * The number of most frequent classes to consider.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 27,
        'endLine' => 27,
        'startColumn' => 5,
        'endColumn' => 21,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'classes' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'name' => 'classes',
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
        'default' => 
        array (
          'code' => '[]',
          'attributes' => 
          array (
            'startLine' => 34,
            'endLine' => 36,
            'startTokenPos' => 60,
            'startFilePos' => 681,
            'endTokenPos' => 64,
            'endFilePos' => 698,
          ),
        ),
        'docComment' => '/**
 * The k most frequent classes.
 *
 * @var list<string>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 34,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 6,
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
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'k' => 
          array (
            'name' => 'k',
            'default' => 
            array (
              'code' => '1',
              'attributes' => 
              array (
                'startLine' => 42,
                'endLine' => 42,
                'startTokenPos' => 81,
                'startFilePos' => 820,
                'endTokenPos' => 81,
                'endFilePos' => 820,
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
            'startLine' => 42,
            'endLine' => 42,
            'startColumn' => 33,
            'endColumn' => 42,
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
 * @param int $k
 * @throws InvalidArgumentException
 */',
        'startLine' => 42,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'currentClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'aliasName' => NULL,
      ),
      'type' => 
      array (
        'name' => 'type',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\DataType',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the data type the strategy handles.
 *
 * @return DataType
 */',
        'startLine' => 57,
        'endLine' => 60,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'currentClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'aliasName' => NULL,
      ),
      'fitted' => 
      array (
        'name' => 'fitted',
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
 * Has the strategy been fitted?
 *
 * @internal
 *
 * @return bool
 */',
        'startLine' => 69,
        'endLine' => 72,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'currentClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'aliasName' => NULL,
      ),
      'fit' => 
      array (
        'name' => 'fit',
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
            'startLine' => 82,
            'endLine' => 82,
            'startColumn' => 25,
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
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Fit the guessing strategy to a set of values.
 *
 * @internal
 *
 * @param list<string> $values
 * @throws InvalidArgumentException
 */',
        'startLine' => 82,
        'endLine' => 96,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'currentClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'aliasName' => NULL,
      ),
      'guess' => 
      array (
        'name' => 'guess',
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
 * Make a guess.
 *
 * @internal
 *
 * @throws RuntimeException
 * @return string
 */',
        'startLine' => 106,
        'endLine' => 115,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'currentClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
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
 * Return the string representation of the object.
 *
 * @internal
 *
 * @return string
 */',
        'startLine' => 124,
        'endLine' => 127,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
        'currentClassName' => 'Rubix\\ML\\Strategies\\KMostFrequent',
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