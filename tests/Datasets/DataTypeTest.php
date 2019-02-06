<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Datasets\DataType;
use PHPUnit\Framework\TestCase;

class DataTypeTest extends TestCase
{
    public function test_determine_type()
    {
        $this->assertEquals(DataType::CONTINUOUS, DataType::determine(2.75));
        $this->assertEquals(DataType::CATEGORICAL, DataType::determine('string'));
        $this->assertEquals(DataType::OTHER, DataType::determine(null));
    }
}