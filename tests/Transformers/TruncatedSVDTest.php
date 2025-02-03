<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\TruncatedSVD;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('tensor')]
#[CoversClass(TruncatedSVD::class)]
class TruncatedSVDTest extends TestCase
{
    protected Blob $generator;

    protected TruncatedSVD $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(
            center: [0.0, 3000.0, -6.0, 25],
            stdDev: [1.0, 30.0, 0.001, 10.0]
        );

        $this->transformer = new TruncatedSVD(2);
    }

    public function testFitTransform() : void
    {
        $this->assertEquals(4, $this->generator->dimensions());

        $this->transformer->fit($this->generator->generate(30));

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(1)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(2, $sample);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
