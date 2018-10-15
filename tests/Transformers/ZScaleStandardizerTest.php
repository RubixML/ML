<?php

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\Transformers\Elastic;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\ZScaleStandardizer;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class ZScaleStandardizerTest extends TestCase
{
    protected $generator;
    
    protected $transformer;

    public function setUp()
    {
        $this->generator = new Blob([0., 3000., -6.], [1., 30., 0.001]);

        $this->transformer = new ZScaleStandardizer(true);
    }

    public function test_build_transformer()
    {
        $this->assertInstanceOf(ZScaleStandardizer::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Elastic::class, $this->transformer);
    }

    public function test_fit_update_transform()
    {
        $this->transformer->fit($this->generator->generate(30));

        $this->transformer->update($this->generator->generate(30));

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->row(0);

        $this->assertCount(3, $sample);
        
        $this->assertEquals(0., $sample[0], '', 3.5);
        $this->assertEquals(0., $sample[1], '', 3.5);
        $this->assertEquals(0., $sample[2], '', 3.5);
    }

    public function test_transform_unfitted()
    {
        $this->expectException(RuntimeException::class);

        $this->generator->generate(1)
            ->apply($this->transformer);
    }
}
