<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/Scoring.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\AnomalyDetectors\Scoring
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-cf7905a05d170fc55b00664d22ffe5d69485be85d5f2d82976273f39afaffb00',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/AnomalyDetectors/Scoring.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\AnomalyDetectors',
    'name' => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
    'shortName' => 'Scoring',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => NULL,
    'attributes' => 
    array (
    ),
    'startLine' => 8,
    'endLine' => 17,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Estimator',
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
      'score' => 
      array (
        'name' => 'score',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 16,
            'endLine' => 16,
            'startColumn' => 27,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the anomaly scores assigned to the samples in a dataset.
 *
 * @param Dataset $dataset
 * @return float[]
 */',
        'startLine' => 16,
        'endLine' => 16,
        'startColumn' => 5,
        'endColumn' => 52,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
        'implementingClassName' => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
        'currentClassName' => 'Rubix\\ML\\AnomalyDetectors\\Scoring',
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