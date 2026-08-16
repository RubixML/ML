<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Strategies/WildGuess.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Strategies\WildGuess
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-c8d9ee570a519bf047d2b428aa5f6cd65c577fcdc32b904ac8c1d1c4309e0e42',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Strategies\\WildGuess',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Strategies/WildGuess.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Strategies',
    'name' => 'Rubix\\ML\\Strategies\\WildGuess',
    'shortName' => 'WildGuess',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Wild Guess
 *
 * Guess a random number somewhere between the minimum and maximum computed by fitting a collection of values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 18,
    'endLine' => 116,
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
      'min' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'name' => 'min',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'int',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 25,
            'endLine' => 25,
            'startTokenPos' => 45,
            'startFilePos' => 536,
            'endTokenPos' => 45,
            'endFilePos' => 539,
          ),
        ),
        'docComment' => '/**
 * The minimum value of the fitted data.
 *
 * @var int|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'max' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'name' => 'max',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'int',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 32,
            'endLine' => 32,
            'startTokenPos' => 59,
            'startFilePos' => 658,
            'endTokenPos' => 59,
            'endFilePos' => 661,
          ),
        ),
        'docComment' => '/**
 * The maximum value of the fitted data.
 *
 * @var int|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 32,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'phi' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'name' => 'phi',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionUnionType',
          'data' => 
          array (
            'types' => 
            array (
              0 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'float',
                  'isIdentifier' => true,
                ),
              ),
              1 => 
              array (
                'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
                'data' => 
                array (
                  'name' => 'null',
                  'isIdentifier' => true,
                ),
              ),
            ),
          ),
        ),
        'default' => 
        array (
          'code' => 'null',
          'attributes' => 
          array (
            'startLine' => 39,
            'endLine' => 39,
            'startTokenPos' => 73,
            'startFilePos' => 818,
            'endTokenPos' => 73,
            'endFilePos' => 821,
          ),
        ),
        'docComment' => '/**
 * A constant that determines the precision of the floating point numbers.
 *
 * @var float|null
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 39,
        'endLine' => 39,
        'startColumn' => 5,
        'endColumn' => 33,
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
        'startLine' => 46,
        'endLine' => 49,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'currentClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
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
        'startLine' => 58,
        'endLine' => 61,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'currentClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
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
            'startLine' => 71,
            'endLine' => 71,
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
 * @param (int|float)[] $values
 * @throws InvalidArgumentException
 */',
        'startLine' => 71,
        'endLine' => 86,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'currentClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Make a continuous guess.
 *
 * @internal
 *
 * @throws RuntimeException
 * @return float
 */',
        'startLine' => 96,
        'endLine' => 103,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'currentClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
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
        'startLine' => 112,
        'endLine' => 115,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Strategies',
        'declaringClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'implementingClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
        'currentClassName' => 'Rubix\\ML\\Strategies\\WildGuess',
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