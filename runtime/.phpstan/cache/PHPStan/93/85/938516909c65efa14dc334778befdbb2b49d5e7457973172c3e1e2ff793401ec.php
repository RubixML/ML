<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Metrics/ProbabilisticAccuracy.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-99bf14d4c7c0a6a14d716f59e7d3d0979baf086f6b927912982f79a82472a7a4',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Metrics/ProbabilisticAccuracy.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
    'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
    'shortName' => 'ProbabilisticAccuracy',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Probabilistic Accuracy
 *
 * This metric comes from the sports betting domain, where it\'s used to measure the accuracy of
 * predictions by looking at the probabilities of class predictions. Accordingly, this metric places
 * additional weight on the "confidence" of each prediction.
 *
 * !!! note
 *     Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 22,
    'endLine' => 65,
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
    ),
    'immediateMethods' => 
    array (
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
        'startLine' => 27,
        'endLine' => 30,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
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
            'startLine' => 39,
            'endLine' => 39,
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
            'startLine' => 39,
            'endLine' => 39,
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
        'startLine' => 39,
        'endLine' => 56,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
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
        'startLine' => 61,
        'endLine' => 64,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\ProbabilisticAccuracy',
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