<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Extractors/Concatenator.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Extractors\Concatenator
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-199470fe40d362f8b59ca1d85a3e1633fff9d66aeeb7ed9d159f7a5a9b0d8a46',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Extractors\\Concatenator',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Extractors/Concatenator.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Extractors',
    'name' => 'Rubix\\ML\\Extractors\\Concatenator',
    'shortName' => 'Concatenator',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Concatenator
 *
 * Combines multiple iterators by concatenating the output of one iterator with the output of
 * the next iterator in the series.
 *
 * @category    Machine Learning
 * @package     Rubix\\ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 17,
    'endLine' => 47,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Extractors\\Extractor',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'iterators' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
        'name' => 'iterators',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'iterable',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * An iterator of iterators.
 *
 * @var iterable<iterable<mixed[]>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 5,
        'endColumn' => 34,
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
          'iterators' => 
          array (
            'name' => 'iterators',
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
            'startLine' => 29,
            'endLine' => 29,
            'startColumn' => 33,
            'endColumn' => 51,
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
 * @param iterable<iterable<mixed[]>> $iterators
 */',
        'startLine' => 29,
        'endLine' => 32,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Extractors',
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
        'currentClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
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
        ),
        'docComment' => '/**
 * Return an iterator for the rows of a data table.
 *
 * @return \\Generator<mixed[]>
 */',
        'startLine' => 39,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => true,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Extractors',
        'declaringClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
        'implementingClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
        'currentClassName' => 'Rubix\\ML\\Extractors\\Concatenator',
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