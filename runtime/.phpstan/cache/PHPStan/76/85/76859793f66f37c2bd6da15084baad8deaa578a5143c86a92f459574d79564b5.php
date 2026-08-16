<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/benchmarks/Persisters/Serializers/GzipNativeBench.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Benchmarks\Persisters\Serializers\GzipNativeBench
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-69d5ea54d6892858c06ea939f6ec7a2da58dd26ed0a8a57e6aebbfdb29b0128a',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'filename' => '/home/andrew/Workspace/Rubix/ML/benchmarks/Persisters/Serializers/GzipNativeBench.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers',
    'name' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
    'shortName' => 'GzipNativeBench',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @Groups({"Serializers"})
 * @BeforeMethods({"setUp"})
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 14,
    'endLine' => 59,
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
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'name' => 'TRAINING_SIZE',
        'modifiers' => 2,
        'type' => NULL,
        'value' => 
        array (
          'code' => '10000',
          'attributes' => 
          array (
            'startLine' => 16,
            'endLine' => 16,
            'startTokenPos' => 43,
            'startFilePos' => 354,
            'endTokenPos' => 43,
            'endFilePos' => 358,
          ),
        ),
        'docComment' => NULL,
        'attributes' => 
        array (
        ),
        'startLine' => 16,
        'endLine' => 16,
        'startColumn' => 5,
        'endColumn' => 42,
      ),
    ),
    'immediateProperties' => 
    array (
      'serializer' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'name' => 'serializer',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var GzipNative
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 21,
        'endLine' => 21,
        'startColumn' => 5,
        'endColumn' => 26,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'persistable' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'name' => 'persistable',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * @var \\Rubix\\ML\\Persistable
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 26,
        'endLine' => 26,
        'startColumn' => 5,
        'endColumn' => 27,
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
        'startLine' => 28,
        'endLine' => 45,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'aliasName' => NULL,
      ),
      'serializeDeserialize' => 
      array (
        'name' => 'serializeDeserialize',
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
 * @revs(10)
 * @Iterations(5)
 * @OutputTimeUnit("milliseconds", precision=3)
 */',
        'startLine' => 53,
        'endLine' => 58,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers',
        'declaringClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'implementingClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
        'currentClassName' => 'Rubix\\ML\\Benchmarks\\Persisters\\Serializers\\GzipNativeBench',
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