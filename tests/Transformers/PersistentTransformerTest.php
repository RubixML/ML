<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Serializers\Native;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\PolynomialExpander;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(PersistentTransformer::class)]
class PersistentTransformerTest extends TestCase
{
    protected function tearDown() : void
    {
        @unlink('test.model');
    }

    #[Test]
    public function base() : void
    {
        $base = new MinMaxNormalizer();

        $transformer = new PersistentTransformer($base, new Filesystem('test.model'));

        $this->assertSame($base, $transformer->base());
    }

    #[Test]
    public function compatibility() : void
    {
        $transformer = new PersistentTransformer(new Pipeline([]), new Filesystem('test.model'));

        $this->assertEquals(DataType::all(), $transformer->compatibility());

        $transformer = new PersistentTransformer(
            new Pipeline([new PolynomialExpander(2), new MinMaxNormalizer()]),
            new Filesystem('test.model')
        );

        $this->assertEquals([DataType::continuous()], $transformer->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $persister = new Filesystem('test.model');

        $transformer = new PersistentTransformer(
            new Pipeline([new MinMaxNormalizer()]),
            $persister,
            new RBX()
        );

        $expected = [
            'base' => new Pipeline([new MinMaxNormalizer()]),
            'persister' => $persister,
            'serializer' => new RBX(),
        ];

        $this->assertEquals($expected, $transformer->params());
    }

    #[Test]
    public function fitAndFitted() : void
    {
        $transformer = new PersistentTransformer(
            new Pipeline([new MinMaxNormalizer(0.0, 1.0)]),
            new Filesystem('test.model')
        );

        $this->assertFalse($transformer->fitted());

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $this->assertTrue($transformer->fitted());
    }

    #[Test]
    public function fitCapturedByTheBaseTransformer() : void
    {
        $base = new MinMaxNormalizer(0.0, 1.0);

        $transformer = new PersistentTransformer($base, new Filesystem('test.model'));

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $this->assertTrue($base->fitted());
    }

    #[Test]
    public function update() : void
    {
        $transformer = new PersistentTransformer(
            new Pipeline([new MinMaxNormalizer(0.0, 1.0)]),
            new Filesystem('test.model')
        );

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $transformer->update(new Unlabeled(samples: [
            [0.0],
            [4.0],
        ]));

        $samples = [[2.0]];
        $transformer->transform($samples);

        $this->assertEqualsWithDelta(0.5, $samples[0][0], 1e-8);
    }

    #[Test]
    public function updateNonElasticBaseTransformer() : void
    {
        $transformer = new PersistentTransformer(
            new OneHotEncoder(),
            new Filesystem('test.model')
        );

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Base Transformer must implement the Elastic interface.');

        $transformer->update(new Unlabeled(samples: [
            ['a'],
            ['b'],
        ]));
    }

    #[Test]
    public function transform() : void
    {
        $transformer = new PersistentTransformer(
            new Pipeline([
                new MinMaxNormalizer(0.0, 1.0),
                new PolynomialExpander(2),
            ]),
            new Filesystem('test.model')
        );

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $samples = [[1.5], [2.5]];
        $transformer->transform($samples);

        $this->assertEqualsWithDelta([[0.25, 0.0625], [0.75, 0.5625]], $samples, 1e-8);
    }

    #[Test]
    public function transformUnfitted() : void
    {
        $transformer = new PersistentTransformer(
            new MinMaxNormalizer(),
            new Filesystem('test.model')
        );

        $samples = [[1.0]];

        $this->expectException(RuntimeException::class);

        $transformer->transform($samples);
    }

    #[Test]
    public function saveAndLoad() : void
    {
        $persister = new Filesystem('test.model');

        $transformer = new PersistentTransformer(
            new Pipeline([
                new MinMaxNormalizer(0.0, 1.0),
                new PolynomialExpander(2),
            ]),
            $persister
        );

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $transformer->save();

        $restored = PersistentTransformer::load($persister);

        $this->assertInstanceOf(PersistentTransformer::class, $restored);
        $this->assertInstanceOf(Pipeline::class, $restored->base());
        $this->assertTrue($restored->fitted());

        $samples = [[1.5], [2.5]];
        $restored->transform($samples);

        $this->assertEqualsWithDelta([[0.25, 0.0625], [0.75, 0.5625]], $samples, 1e-8);
    }

    #[Test]
    public function saveAndLoadSingleTransformer() : void
    {
        $persister = new Filesystem('test.model');

        $transformer = new PersistentTransformer(new MinMaxNormalizer(0.0, 1.0), $persister);

        $transformer->fit(new Unlabeled(samples: [
            [1.0],
            [2.0],
            [3.0],
        ]));

        $transformer->save();

        $restored = PersistentTransformer::load($persister);

        $this->assertInstanceOf(MinMaxNormalizer::class, $restored->base());

        $samples = [[1.0], [3.0]];
        $restored->transform($samples);

        $this->assertEqualsWithDelta([[0.0], [1.0]], $samples, 1e-8);
    }

    #[Test]
    public function saveAndLoadEmptyPipeline() : void
    {
        $persister = new Filesystem('test.model');

        $transformer = new PersistentTransformer(new Pipeline([]), $persister);

        $transformer->save();

        $restored = PersistentTransformer::load($persister);

        $samples = [[1.0, 2.0]];
        $restored->transform($samples);

        $this->assertSame([[1.0, 2.0]], $samples);
    }

    #[Test]
    public function saveAndLoadWithCustomSerializer() : void
    {
        $persister = new Filesystem('test.model');

        $transformer = new PersistentTransformer(
            new Pipeline([new MinMaxNormalizer()]),
            $persister,
            new Native()
        );

        $transformer->save();

        $this->assertStringNotContainsString("\251RBX", $persister->load()->data());

        $restored = PersistentTransformer::load($persister, new Native());

        $this->assertEquals(
            [new MinMaxNormalizer()],
            $restored->base()->params()['transformers']
        );
    }

    #[Test]
    public function saveDoesNotPersistTheStorageCoordinates() : void
    {
        $transformer = new PersistentTransformer(
            new Pipeline([new MinMaxNormalizer()]),
            new Filesystem('test.model')
        );

        $encoding = (new Native())->serialize($transformer->base());

        $this->assertStringNotContainsString('test.model', $encoding->data());
    }

    #[Test]
    public function loadRejectsNonStateful() : void
    {
        $persister = new Filesystem('test.model');

        $persister->save((new Native())->serialize(new GaussianNB()));

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Persisted object must implement the Stateful interface.');

        PersistentTransformer::load($persister, new Native());
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $transformer = new PersistentTransformer(
            new Pipeline([new MinMaxNormalizer()]),
            new Filesystem('test.model')
        );

        $this->assertStringStartsWith('Persistent Transformer', (string) $transformer);
    }
}
