<?php

namespace Rubix\Tests\Extractors;

use Rubix\ML\Extractors\Extractor;
use Rubix\ML\Extractors\ImagePatchDescriptor;
use Rubix\ML\Extractors\Descriptors\TextureHistogram;
use PHPUnit\Framework\TestCase;

class ImagePatchDescriptorTest extends TestCase
{
    protected $extractor;

    protected $samples;

    protected $descriptors;

    public function setUp()
    {
        $this->samples = [
            imagecreatefromjpeg(__DIR__ . '/space.jpg'),
        ];

        $this->descriptors = [
            new TextureHistogram(),
        ];

        $this->extractor = new ImagePatchDescriptor($this->descriptors, [8, 8], [4, 4], 'gd');
    }

    public function test_build_count_vectorizer()
    {
        $this->assertInstanceOf(ImagePatchDescriptor::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
    }

    public function test_num_patches()
    {
        $this->assertEquals(4, $this->extractor->numPatches());
    }

    public function test_transform_dataset()
    {
        $this->extractor->fit($this->samples);

        $samples = $this->extractor->extract($this->samples);

        $this->assertEquals([
            [49.5625, 267.42664930555566, 0.01500710350411747, -0.7975132983112356,
            57.22916666666667, 465.3849826388888, 0.292377121740332, -1.0110518843270149,
            61.77083333333333, 223.24609374999997, 0.024777058396068665, -0.9761468969499858,
            32.666666666666664, 184.5, 0.34731531590168385, -1.2421806211437605],
        ], $samples);
    }
}
