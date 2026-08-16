<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Clusterers/DBSCANBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Clusterers\DBSCANBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-591a4c33dcd5e3ea16a51dd059d48bc4c9ebbe16e8c376234a920d969baaa564',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Clusterers/DBSCANBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Clusterers',
    'name' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
    'shortName' => 'DBSCANBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Clusterers"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 13,
    'endLine' => 49,
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
      'TESTING_SIZE' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'name' => 'TESTING_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 15,
            'endLine' => 15,
            'startTokenPos' => 38,
            'startFilePos' => 287,
            'endTokenPos' => 38,
            'endFilePos' => 291,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 15,
        'endLine' => 15,
        'startColumn' => 5,
        'endColumn' => 41,
      ),
    ),
    'immediateProperties' => 
    array (
      'testing' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'name' => 'testing',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var \\Rubix\\ML\\Datasets\\Labeled
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 20,
        'endLine' => 20,
        'startColumn' => 5,
        'endColumn' => 23,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'estimator' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'name' => 'estimator',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var DBSCAN
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 25,
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
        'startLine' => 27,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'aliasName' => NULL,
      ),
      'predict' => 
      array (
        'name' => 'predict',
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
        'endLine' => 48,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Clusterers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Clusterers\\DBSCANBench',
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