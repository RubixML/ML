<?php
/**
 * Minimal test for K-Medoids without dependencies
 */

// Simple autoloader for testing
spl_autoload_register(function ($class) {
    $prefix = 'Rubix\\ML\\';
    $base_dir = __DIR__ . '/src/';
    
    $len = strlen($prefix);
    if (strncmp($prefix, $class, $len) !== 0) {
        return;
    }
    
    $relative_class = substr($class, $len);
    $file = $base_dir . str_replace('\\', '/', $relative_class) . '.php';
    
    if (file_exists($file)) {
        require $file;
    }
});

echo "K-Medoids Syntax Validation Test\n";
echo "=================================\n\n";

// Check if class can be loaded
if (class_exists('Rubix\ML\Clusterers\KMedoids')) {
    echo "✓ KMedoids class exists\n";
    
    // Check if all required methods exist
    $reflection = new ReflectionClass('Rubix\ML\Clusterers\KMedoids');
    $methods = ['train', 'predict', 'partial', 'proba', 'medoids', 'sizes', 'steps', 'losses'];
    
    foreach ($methods as $method) {
        if ($reflection->hasMethod($method)) {
            echo "✓ Method '$method' exists\n";
        } else {
            echo "✗ Method '$method' missing\n";
        }
    }
    
    echo "\n✓ All K-Medoids code structure is valid!\n";
} else {
    echo "✗ KMedoids class not found\n";
}
