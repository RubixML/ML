<?php

namespace Rubix\Tests\Extractors;

use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\PixelEncoder;
use PHPUnit\Framework\TestCase;

class PixelEncoderTest extends TestCase
{
    protected $extractor;

    protected $samples;

    public function setUp()
    {
        $this->samples = [
            imagecreatefromjpeg(__DIR__ . '/space.jpg'),
        ];

        $this->extractor = new PixelEncoder([3, 3], true, 0, 'gd');
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(PixelEncoder::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
    }

    public function test_transform_dataset()
    {
        $this->extractor->fit($this->samples);

        $samples = $this->extractor->extract($this->samples);

        $this->assertEquals([
            [22, 35, 60, 53, 66, 102, 29, 44, 73, 36, 49, 79, 45, 57, 89, 21,
            32, 56, 44, 53, 85, 43, 49, 75, 12, 18, 34],
        ], $samples);
    }
}
