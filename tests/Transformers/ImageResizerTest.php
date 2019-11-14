<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\ImageResizer;
use PHPUnit\Framework\TestCase;

class ImageResizerTest extends TestCase
{
    protected $transformer;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Unlabeled::quick([
            [imagecreatefromjpeg(__DIR__ . '/../space.jpg')],
        ]);

        $this->transformer = new ImageResizer(32, 32, 'gd');
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(ImageResizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    public function test_transform()
    {
        $this->dataset->apply($this->transformer);

        $sample = $this->dataset->sample(0);
    
        $image = $sample[0];

        $this->assertEquals(32, imagesx($image));
        $this->assertEquals(32, imagesy($image));
    }
}
