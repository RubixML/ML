<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Metrics/MedianAbsoluteError.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-51817c9ab924aa5e258090af920050699e73d1c4d09368b123e368eaf1b5d360',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Metrics/MedianAbsoluteError.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
    'name' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
    'shortName' => 'MedianAbsoluteError',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Median Absolute Error
 *
 * Median Absolute Error (MAD) is a robust measure of error, similar to MAE, that ignores
 * highly erroneous predictions. Since MAD is a robust statistic, it works well even when
 * used to measure non-normal distributions.
 *
 * > **Note:** In order to maintain the convention of *maximizing* validation scores,
 * this metric outputs the negative of the original score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 24,
    'endLine' => 86,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\CrossValidation\\Metrics\\Metric',
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
 * Return a tuple of the min and max output value for this metric.
 *
 * @return \\Rubix\\ML\\Tuple{float,float}
 */',
        'startLine' => 31,
        'endLine' => 34,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'aliasName' => NULL,
      ),
      'compatibility' => 
      array (
        'name' => 'compatibility',
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
 * The estimator types that this metric is compatible with.
 *
 * @internal
 *
 * @return list<EstimatorType>
 */',
        'startLine' => 43,
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'aliasName' => NULL,
      ),
      'score' => 
      array (
        'name' => 'score',
        'parameters' => 
        array (
          'predictions' => 
          array (
            'name' => 'predictions',
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
            'startLine' => 58,
            'endLine' => 58,
            'startColumn' => 27,
            'endColumn' => 44,
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
            'startLine' => 58,
            'endLine' => 58,
            'startColumn' => 47,
            'endColumn' => 59,
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
 * Score a set of predictions.
 *
 * @param list<int|float> $predictions
 * @param list<int|float> $labels
 * @throws \\Rubix\\ML\\Exceptions\\InvalidArgumentException
 * @return float
 */',
        'startLine' => 58,
        'endLine' => 73,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
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
        'startLine' => 82,
        'endLine' => 85,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Metrics',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Metrics\\MedianAbsoluteError',
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