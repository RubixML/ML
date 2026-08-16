<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/AnomalyDetectors/RobustZScoreBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\AnomalyDetectors\RobustZScoreBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-d2453ff9298e20df6e94c22c199794f92a6cc18855ce254b05b6aebb42b12d34',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/AnomalyDetectors/RobustZScoreBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors',
    'name' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
    'shortName' => 'RobustZScoreBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"AnomalyDetectors"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 14,
    'endLine' => 51,
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
      'TRAINING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'name' => 'TRAINING_SIZE',
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
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 16,
            'endLine' => 16,
            'startTokenPos' => 45,
            'startFilePos' => 353,
            'endTokenPos' => 45,
            'endFilePos' => 357,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 16,
        'endLine' => 16,
        'startColumn' => 5,
        'endColumn' => 46,
      ),
      'TESTING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'name' => 'TESTING_SIZE',
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
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 18,
            'endLine' => 18,
            'startTokenPos' => 58,
            'startFilePos' => 400,
            'endTokenPos' => 58,
            'endFilePos' => 404,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 18,
        'endLine' => 18,
        'startColumn' => 5,
        'endColumn' => 45,
      ),
    ),
    'immediateProperties' => 
    array (
      'training' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'name' => 'training',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Datasets\\Labeled',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 32,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'testing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'name' => 'testing',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Datasets\\Labeled',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 22,
        'endLine' => 22,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'estimator' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'name' => 'estimator',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\AnomalyDetectors\\RobustZScore',
            'isIdentifier' => false,
          ),
        ),
        'default' => NULL,
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 5,
        'endColumn' => 38,
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
      'setUp' => 
      array (
        'name' => 'setUp',
        'parameters' => 
        array (
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
        'docComment' => NULL,
        'startLine' => 26,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'aliasName' => NULL,
      ),
      'trainPredict' => 
      array (
        'name' => 'trainPredict',
        'parameters' => 
        array (
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
 * @Subject
 * @Iterations(5)
 * @OutputTimeUnit("seconds", precision=3)
 */',
        'startLine' => 45,
        'endLine' => 50,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\AnomalyDetectors\\RobustZScoreBench',
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