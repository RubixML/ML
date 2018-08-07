<?php

namespace Rubix\Tests\Extractors;

use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\RawPixelEncoder;
use PHPUnit\Framework\TestCase;

class RawPixelEncoderTest extends TestCase
{
    protected $extractor;

    protected $samples;

    public function setUp()
    {
        $this->samples = [
            imagecreatefromjpeg(__DIR__ . '/space.jpg'),
        ];

        $this->extractor = new RawPixelEncoder([3, 3], true, 'gd');
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(RawPixelEncoder::class, $this->extractor);
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
