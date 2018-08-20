<?php

namespace Rubix\Tests\Extractors\Descriptors;

use Rubix\ML\Extractors\Descriptors\Descriptor;
use Rubix\ML\Extractors\Descriptors\AverageColor;
use PHPUnit\Framework\TestCase;

class AverageColorTest extends TestCase
{
    protected $descriptor;

    protected $patch;

    public function setUp()
    {
        $this->patch = [
            [[10, 18, 36], [18, 30, 53], [30, 36, 77], [35, 51, 84]],
            [[14, 22, 43], [24, 39, 66], [40, 56, 91], [39, 55, 89]],
            [[17, 27, 48], [29, 43, 72], [43, 58, 93], [37, 49, 79]],
            [[20, 31, 55], [32, 47, 76], [52, 63, 97], [64, 72, 109]],
        ];

        $this->descriptor = new AverageColor();
    }

    public function test_build_descriptor()
    {
        $this->assertInstanceOf(AverageColor::class, $this->descriptor);
        $this->assertInstanceOf(Descriptor::class, $this->descriptor);
    }

    public function test_describe_patch()
    {
        $output = $this->descriptor->describe($this->patch);

        $this->assertEquals([63, 87, 146], $output);
    }
}
