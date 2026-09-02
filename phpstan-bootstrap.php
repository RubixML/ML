<?php

declare(strict_types=1);

// NumPower dtype names are used as constants across the codebase.
// During static analysis these constants may not exist, so we define them here
// to prevent undefined constant errors in PHPStan.
foreach ([
    'float16',
    'float32',
    'float64',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'bool',
] as $dtype) {
    if (!defined($dtype)) {
        define($dtype, $dtype);
    }
}
