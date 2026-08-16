<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Metrics/TopKAccuracy.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\CrossValidation\Metrics\TopKAccuracy
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-43694d76b31359ff58891fcb9b1d4d89e50c7158beabe541aacb248cddf6ee78',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Metrics/TopKAccuracy.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
    'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
    'shortName' => 'TopKAccuracy',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Top K Accuracy
 *
 * Top K Accuracy looks at the k classes with the highest predicted probabilities when
 * calculating the accuracy score. If one of the top k classes matches the ground-truth,
 * then the prediction is considered accurate.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 25,
    'endLine' => 101,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticMetric',
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
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
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
 * The number of classes with the highest predicted probability to consider.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 32,
        'endLine' => 32,
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
              'code' => '3',
              'attributes' => 
              array (
                'startLine' => 38,
                'endLine' => 38,
                'startTokenPos' => 85,
                'startFilePos' => 936,
                'endTokenPos' => 85,
                'endFilePos' => 936,
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
            'startLine' => 38,
            'endLine' => 38,
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
        'startLine' => 38,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'aliasName' => NULL,
      ),
      'range' => 
      array (
        'name' => 'range',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Tuple',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * {@inheritDoc}
 */',
        'startLine' => 51,
        'endLine' => 54,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'aliasName' => NULL,
      ),
      'score' => 
      array (
        'name' => 'score',
        'parameters' => 
        array (
          'probabilities' => 
          array (
            'name' => 'probabilities',
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
            'startLine' => 63,
            'endLine' => 63,
            'startColumn' => 27,
            'endColumn' => 46,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'labels' => 
          array (
            'name' => 'labels',
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
            'startLine' => 63,
            'endLine' => 63,
            'startColumn' => 49,
            'endColumn' => 61,
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
            'name' => 'float',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the validation score of a set of probabilities with their ground-truth labels.
 *
 * @param list<array<string|int,float>> $probabilities
 * @param list<string|int> $labels
 * @return float
 */',
        'startLine' => 63,
        'endLine' => 92,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
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
 * {@inheritDoc}
 */',
        'startLine' => 97,
        'endLine' => 100,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\TopKAccuracy',
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