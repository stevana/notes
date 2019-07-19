---
title: Model-based testing meets test doubles and consumer-driven contracts
author: Stevan Andjelkovic
date: Fr 19. Jul 09:42:14 CEST 2019
---

I'd like to share some thoughts on how to improve testing of software by combine
ideas from _model-based testing_ with _test doubles_ (in particular _fakes_) and
_consumer-driven contracts_.

The end goal is to derive fakes from state machine models and be able to verify
that the fakes accurately represent the real components using model-based
testing. By taking this idea to its extream we end up with _simulation testing_,
but with higher assurances as we have made sure that the fake components used as
part of the simulation act like the real components.

If you aren't familiar with all the emphasised words above, don't worry: I'll
explain them each in turn. If you on the otherhand feel comfortable with these
concepts already, then feel free to skim or even skip the next three sections.

I should also mention that I'll be using Haskell, but I've tried to keep it
simple and implement almost everything from first principle so that people not
familiar with Haskell can still follow along.

### Test double

A [test double](https://www.martinfowler.com/bliki/TestDouble.html) is a
component that is used in place of the real production component for testing
purposes. Martin Fowler uses the analogy between a test double and a stunt
double in movies, both stand in for the real thing (component or actor). There
are different kinds of test doubles, all explained in the above link, we shall
focus on so called _fakes_. A fake behaves like the real thing, but take some
shortcut making them unsuitable for production use. Fowler gives the example of
an in-memory database. An in-memory database behaves like a real database, but
doesn't save to disk (shortcut) and so in case of, for example, a power failure
it will lose all data (not suitable for production).

There are several reasons why you might want to use test doubles in your tests.
It could be that setting up and interacting with the real component is slow,
e.g. a database. It could be that interacting with the real component is
non-deterministic, which could result in flaky tests especially when you depend
on many non-deterministic componenets, e.g. an external web service or
networking in general. It could be that the real component doesn't exist yet,
e.g. it's some other team's responsibility to deliver it, but you still want to
be able to run your component and write your tests which both depend on the yet
non-existing component.

In order to be able to swap in your test doubles instead of the real components
when you do testing you will need some sort of interface-based architecture.
I'll will not discuss how to do this here as this is programming language
dependent and a big topic in its own right, even given a specific language. See
discussion on MTL vs records of functions vs free monads in functional
programming circles, or dependency injection in object oriented programming
circles.

### (Consumer-driven) contract testing

The danger of using test doubles is that they might not actually behave like the
real production component. It could be because of some assumptions we made about
the real component when we wrote the test double that were not true, or perhaps
they were true but the real component changed and we forgot to update the test
double.

This is where [contract tests](https://martinfowler.com/bliki/ContractTest.html)
come in. A contract test basically compares the output of test double with the
output of the real component. Often the real component, say a web service, is
maintained by another team than the one you're working on which means the API
can change in a breaking way. By putting the contract test inside the other
team's test suite they will be alerted if they break the a part of the API which
your tests doubles rely on. These kind of contracts which are written by the
consumer of the API, but put in the producer's test suite are called
[consumer-driven
contracts](https://martinfowler.com/articles/consumerDrivenContracts.html).

(It appears that contract tested fakes are sometimes refered to as [verified
fakes](https://pythonspeed.com/articles/verified-fakes/), but the terminology
doesn't seem to be well established.)

### Model-based testing

[Model-based testing](https://en.wikipedia.org/wiki/Model-based_testing) is
perhaps less well known than test doubles and contracts. For sure it's less
practiced at least.

The basic idea is that implement a simplified model of the real component that
is supposed to be tested and then generate test cases that can be run both
against the model and the real system and then compare if the outputs match up.
It's a little bit like the scientific method: you come up with a model for some
part of reality, you do some experiments, collect measurements and then check
that you model correctly predicated the outcomes.

If your model, the real component or both are wrong you get a counterexample
which is a useful starting point for debugging. In the scientific method analogy
the real component, i.e. reality or nature, can't be wrong of course, but in
software where programmers are the gods that will happen for sure.

I first encountered model-based testing when reading papers about Quviq's
proprietary Erlang version of QuickCheck. This proprietary version of QuickCheck
extends the original Haskell version of QuickCheck with model-based testing
support where the models are captured by state machines.

For those not familiar with QuickCheck, it's a library for so called
_property-based testing_. I don't want to explain property-based testing in
isolation from model-based testing, because I feel the two belong together and
it's a shame that we got all these QuickCheck
[clones](https://en.wikipedia.org/wiki/QuickCheck) that only do property-based
testing and have no support for the model-based testing part.

So I'll now try to explain property-based and model-based testing at the same
time by means of a filesystem example.

Recall that the _model_ in model-based testing is analogues to a scientific
theory/model that can predict the outcome of an experiment. Now let's imagine we
want to test if `writeFile` (write to a file) and `readFile` (read from a file)
behave like we expect them to. These two, impure or stateful, functions are part
of the Haskell standard library (or `Prelude`) where their type signatures are
as follows:

```haskell
  writeFile :: FilePath -> String -> IO ()
  readFile  :: FilePath -> IO String
```

Pure functions, like say `not :: Bool -> Bool` or `(+ 1) :: Int -> Int` and
unlike `writeFile` and `readFile`, can be thought of black boxes where you feed
in some input and you get an output:

```
                +--------------------------------+
                |                                |
    Input       |        Pure function           |    Output
 -------------->|                                |--------------->
                |                                |
                +--------------------------------+
```

Note that pure functions always give the same output if you feed it the same
input. When we deal with impure or stateful functions this is not the case. The
black box looks more like this:

```
                +--------------------------------+
                |                                |
                |       +-----------------+      |
                |   +-->|   State/Model   |--+   |
                |   |   |                 |  |   |
 Input/Command  |   |   +-----------------+  |   | Output/Response
                |   |                        v   |
 -------------->|---|------------------------|-->|--------------->
                |                                |
                +--------------------------------+
```

Where the output depends on the history of inputs. Model-based testing is
essentially coming up with a representation of the state or the model, such that
we can predict the outputs/responses of a sequence of inputs/commands.

In our example, let's say we expect that reading from a file should return the
contents that we most recently wrote to it. In order to keep things simple,
let's also assume that reading from a file that has not been written to, causes
a file not found exception to be thrown. The above diagram can be captured with
the following function:

```haskell
  data Command
    = WriteFile FilePath String
    | ReadFile FilePath

  data Response
    = WriteResp ()
    | ReadResp String
    | Exception String

  sequenceBasedSpec :: [Command] -> [Response]
  sequenceBasedSpec = go []
    where
    go :: [Command] -> [Command] -> [Response]
    go _hist []           = []
    go hist  (cmd : cmds) = case cmd of
      WriteFile fp s -> WriteResp () : go (cmd : hist) cmds
      ReadFile  fp   -> case lookupWriteTo fp hist of
        Nothing -> Exception ("File not found: " ++ fp) : go (cmd : hist) cmds
        Just s  -> ReadResp s : go (cmd : hist) cmds

    lookupWriteTo :: FilePath -> [Command] -> Maybe String
    lookupWriteTo fp = go
      where
        go []           = Nothing
        go (cmd : cmds) = case cmd of
          WriteFile fp' s | fp' == fp -> Just s
                          | otherwise -> go cmds
          ReadFile _fp -> go cmds
```

From a list of inputs/commands, we get a list of outputs/responses back. Note
that the above function is a pure model, it doesn't do any filesystem
interaction. Using this function we can predict the outcome of a real filesystem
interaction though (assuming the real filesystem behaves according to the
assumtions that we made in the model).

For example, the sequence of inputs:

```haskell
    [ WriteFile "/tmp/file1" "foo"
    , ReadFile "/tmp/file1"
    , ReadFile "/tmp/file2"
    ]
```

Returns the following sequence of outputs:

```haskell
    [ WriteResp ()
    , ReadResp "foo"
    , Exception "File not found: /tmp/file2"
    ]
```

Now that we have our model, we are ready to start experimenting and checking
that the model agrees with the real filesystem. In order to do this we need a
couple of things:

  1. A way to generate a random sequence of inputs (a statistically valid
     sample);
  2. A way to turn our randomly generated input sequence into real filesystem
     interactions (this enables us to compare the output of the model with the
     output of the real system);
  3. A clean up function that resets our testing environment between experiments
     (otherwise the system can degrade over time and introduce bias).

Let's start with how to generate random sequences of commands.

When we humans want to introduce a source of randomness we like to flip coins or
throw dice. One way we could generate a random command is to first decide if
it's a read or a write to a file by flipping a coin. For both reading and
writing we then need to decide a file path to the actual file. If we pick the
file path completely arbitrarily we risk of never actually reading from a file
that we wrote to, so instead we will pick from three hard code three paths.
Finally, in the case of writing to a file we also need to randomly generate a
string to write to the file. One way to do this would be to throw a die that
decides the length of the string, and then throw that many 128-sided (or
ASCII-sided) dice to determine the characters. We most certainly want to account
for empty and unicode strings as well, but I hope you get the idea.

Following the above description and using Haskell's
[`System.Random`](https://hackage.haskell.org/package/random/docs/System-Random.html)
module to flip coins and throw dice, we can implement generation of commands. If
we were to do so we would quickly find that certain combinators are useful to
reduce boilerplate. One of the things that the propery-based testing library
[QuickCheck](https://hackage.haskell.org/package/QuickCheck) does is to provide
such
[combinators](https://hackage.haskell.org/package/QuickCheck/docs/Test-QuickCheck-Gen.html).

Here's how we can generate commands using said combinators:

```haskell
  filePaths :: [FilePath]
  filePaths = ["/tmp/file1", "/tmp/file2", "/tmp/file3"]

  genCommand :: Gen Command
  genCommand = oneof
    [ genReadFile
    , genWriteFile
    ]
    where
      genFilePath :: Gen FilePath
      genFilePath = elements filePaths -- Pick one of the hardcoded filepaths.

      genReadFile :: Gen Command
      genReadFile = do
        fp <- genFilePath
        return (ReadFile fp)

      -- (We can also write this function, using applicative functors, in the
      -- following way: genReadFile = ReadFile <$> genFilePath).

      genWriteFile :: Gen Command
      genWriteFile = do
        fp <- genFilePath
        s  <- arbitrary :: Gen Sting
        return (WriteFile fp s)
      -- (or: genWriteFile = WriteFile <$> genFilePath <*> arbitrary)

    genCommands :: Gen [Command]
    genCommands = listOf genCommand
```

The QuickCheck library also provides us with a way to `sample` our generators to
see what kind of things they generate:

```
ghci> import Test.QuickCheck
ghci> sample genCommands
[...] -- XXX
```

Recall that a `Command` is simply a representation of a filesystem interaction,
to actually perform the interaction we interpret the command inside the `IO`
monad.

```haskell
  interpret :: Command -> IO Response
  interpret (WriteFile fp s) = fmap WriteResp (writeFile fp s) -- XXX: catch
  interpret XXX:...

  interpretMany :: [Command] -> IO [Response]
  interpretMany = traverse interpret
```

In order to be able to make repeated experiments without earlier interactions
interfering with new experiments we need to remove the files that maybe be
involved in an experiment.

```haskell
  cleanUp :: IO ()
  cleanUp = mapM_ removePathForcibly filePaths
```

Finally, we can put everything together and write our property. It can be
helpful to think of a property as a recipe for how to generate test cases.

```haskell
  prop_sequenceBased :: Property
  prop_sequenceBased = forAll genCommands $ \cmds -> monadicIO $ do
    liftIO cleanUp
    resps <- liftIO (interpretMany cmds)
    return (resps === sequenceBasedSpec cmds)
```

This property says, for all commands generated by the generator that we defined
above, if we first clean up and then interpret said commands the list of
responses returned from the interpretation is the same as list of responses
returned by the specfication.

If we pass the above property to `QuickCheck`, then `QuickCheck` will by default
generate 100 random lists of commands, interpret them and compare the responses
with the specification. Here's how it looks inside a REPL:

```
ghci> quickCheck prop_sequenceBased
-- XXX
```

If there would have been a bug in the specification, then `QuickCheck` would
have given us a counterexample instead:

```
ghci> quickCheck prop_sequenceBased True?
-- XXX
```

That's pretty much the idea of model-based testing in a property-based fashion.

One thing we can do at this point is refactor things a bit to make the model
more explicit.


```haskell
  type Model = Map FilePath String

  initModel :: Model
  initModel = Map.empty
```

We can also refactor `sequenceBasedSpec :: [Command] -> [Response]` which acts
on whole lists of commands into two smaller functions that act on individual
commands given some model.

```haskell
  transition :: Model -> Command -> Model
  transition = XXX

  spec :: Model -> Command -> Response
  spec = XXX
```

Note that using these two smaller functions we can reimplement
`sequenceBasedSpec`.

Our property has to be rewritten a little bit as well.


```haskell
  prop_modelBased :: Property
  prop_modelBased = forAll genCommands $ \cmds -> monadicIO $ do
    liftIO cleanUp
    go initModel cmds
    where
      go :: Model -> [Command] -> PropertyM IO Bool
      go _model []           = return True
      go model  (cmd : cmds) = do
        resp <- liftIO (interpret cmd)
        if resp === spec model cmd
        then go (transition model cmd) cmds
        else return False -- XXX: counterexample?
```

Specification of the form `[Command] -> [Response]`, and `Model` together with
`initModel`, `transition` and `spec` are called functional specifications. The
former is sometimes refered to as a black box, sequence- or
[trace-based](https://doi.org/10.1007/3-540-08934-9_80) specification, while the
latter is a state machine specification. The nice thing about sequence-based
specifications is that they require very little new syntax to explain to
somebody not familiar with formal specification. People seem to be more
comfortable thinking in lists of commands as input (or scenarios or use cases)
producing some outputs, than state machines. There's also a mechanical way to
enumerate all sequences of commands which forces you to consider all corner
cases. Furthermore there's also algorithms for refactoring sequence based
specifications, as well as algorithms for transforming a sequence-based
specification into a state machine one. XXX: add links to references

Functions are deterministic, so we cannot specify non-deterministic behaviour
this way. For example if we have a command that flips a coin, then there are two
possible responses where as `spec` only lets us associate one response for each
command. However we can easily generalise `spec` to `relSpec :: Model -> Command
-> Set Response`, this would give us a relational specification rather than a
functional one. In our test we would then check that `Set.member resp (relSpec
model cmd)` rather than `resp === spec model cmd`.

What other useful extensions to this idea are there? Sometimes it's useful to
only allow certain commands to be issued in certain states, e.g. we can only
close a file handle if it's open. It's easy to achieve this by means of a
pre-condition function of type `Model -> Command -> Bool`. One thing to keep in
mind is that in case of failure, when test case minimisation (shrinkng) is
triggered we need to minimise while respecting the pre-conditions.

Another extension that is commonly needed is symbolic references. Imagine we
wanted to extend our filesystem example with `openFile :: FilePath -> IO Handle`
which returns a file handle, that is later used for reading (with `hGetLine ::
Handle -> IO String`) or writing (with `hPutStrLn :: Handle -> String -> IO
()`). The problem is that when we generate commands, we want subsequent reads
and writes to be able to use previously opened file handles, that's what
symbolic references allow us to do -- bind the result of `openFile` to a
symbolic reference, use the reference in subsequent reads and writes. While
interpreting the commands into real filesystem interactions we need to replace
the symbolic references with concrete ones, i.e. real file handles. We do this
by maintaining a map between symbolic and concrete references as we interpret,
and substitute all symbolic reference for concrete ones using the map before
interpreting, etc.

Perhaps the most interesting extension of state machine testing is that we can
concurrent/parellel testing for free via
[linearisability](https://en.wikipedia.org/wiki/Linearizability).
Linearisability essentially says that if we can find a sequential interleaving
(one thread) of our concurrent execution of commands (many threads) that
respects the sequential specification, then the concurrent execution is correct.
This is a very nice result from the classic
[paper](https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf) that can be used
to find race conditions and other problems with concurrent systems.

For more comprehensive libraries which combine property-based and model-based
testing, and have the above discussed extensions, see:

  * [quickcheck-state-machine](https://github.com/advancedtelematic/quickcheck-state-machine)
    written in Haskell by me and others, also has support for race condition or
    parallel testing;
  * Erlang QuickCheck, [eqc](http://quviq.com/documentation/eqc/), the first
    property based testing library to have support for state machine model-based
    testing (closed source);
  * The Erlang library [PropEr](https://github.com/manopapad/proper) is
    *eqc*-inspired, open source, and has support for state machine
    [testing](http://propertesting.com/);
  * The Haskell library
    [Hedgehog](https://github.com/hedgehogqa/haskell-hedgehog), also has support
    for state machine based testing;
  * [ScalaCheck](http://www.scalacheck.org/), likewise has support for state
    machine based
    [testing](https://github.com/rickynils/scalacheck/blob/master/doc/UserGuide.md#stateful-testing)
    (no parallel testing);
  * The Python library
    [Hypothesis](https://hypothesis.readthedocs.io/en/latest/), also has support
    for state machine based
    [testing](https://hypothesis.readthedocs.io/en/latest/stateful.html) (no
    parallel testing).

(If any of the above is incorrect or if you know of other libraries
property-based testing libraries with model-based testing support, please open
an [issue](https://github.com/advancedtelematic/quickcheck-state-machine/issues)
saying that the section on "Other similar libraries" in the
[README](https://github.com/advancedtelematic/quickcheck-state-machine#readme)
is incorrect or can be improved.)

Finally it should be pointed out that there are [other
forms](http://mit.bme.hu/~micskeiz/pages/modelbased_testing.html) of model-based
testing that don't make use of property-based testing.

### Combining the ideas

Having explained test doubles, contracts and model-based testing we can finally
start talking about how we can combine these ideas.

The basic idea is to derive a test double fake from the state machine model. The
perhaps simplest way we can do this using a global variable (`IORef` in
Haskell) that keeps track of the state/model:

```haskell
fileSystemIORef :: IORef Model
fileSystemIORef = newIORef initModel

writeFileFake :: FilePath -> String -> IO ()
writeFileFake fp s = 
  atomicModifyIORef fileSystemIORef $ \fs ->
    let 
      fs' = transition fs (WriteFile fp s)
    in
      (fs', fs')

readFileFake :: FilePath -> IO String
readFileFake = ...

```

We have already tested the state machine model against the real filesystem, so
we have a contract test that the fake filesystem interaction is a faithful
representation of the real.

Apart from the already mentioned benefits of using test double fakes, namely
speed and determinicity, there's also the the benefit of fault injection being
easier.

```haskell
data FileSystem = FileSystem
  { fileSystem :: Model
  , faults     :: Bool
  }

toggleFaults :: IO ()
toggleFaults = atomicModifyIORef fileSystemIORef $ \fs ->
  (fs { faults = not (faults fs) }, ())

```

With this in place it's easier to test that functions that rely on filesystem
interaction do the right thing in presence of filesystem failures. In particular
this will help us detect places where error handling logic is incorrect (which
by the way is a big
[source](https://blog.acolyer.org/2016/10/06/simple-testing-can-prevent-most-critical-failures/)
of bug). We can at this point also extend the specification to handle fault
injection, and make sure that our fake fault injection is faithfully
representing real fault injection which can be achived by, for example,
[libfiu](https://blitiri.com.ar/p/libfiu/).

Another benefit of this approach is that it lets us parallelise development
work. For example if one team is developing component and another team depends
on the first component, then we could give the second team a state machine model
fake of the first component, and later verify that the fake matches the real
component once the first team is done developing it.

XXX: Example, web service
```haskell
newtype Counter = Counter { count :: Natural }
  deriving Num

initCounter :: Counter
initCounter = 0

data Command resp where
  Inc :: Natural -> Command ()
  Get :: Command Natural
deriving instance Show (Command resp)

transition :: Counter -> Command resp -> Counter
transition c (Inc n) = c + Counter n
transition c Get     = c

spec :: Counter -> Command resp -> resp
spec _c Inc {} = ()
spec c  Get    = count c

counter :: IORef Counter
counter = unsafePerformIO (newIORef initCounter)
{-# NOINLINE counter #-}

command :: Command resp -> IO resp
command cmd = do
  c' <- atomicModifyIORef counter $ \c ->
           let
             c' = transition c cmd
           in
             (c', c')
  return (spec c' cmd)

incM :: Natural -> IO ()
incM = command . Inc

getM :: IO Natural
getM = command Get

type API =
  "inc" :> Capture "natural" Natural :> Post '[JSON] () :<|>
  "get" :> Get  '[JSON] Natural

api :: Proxy API
api = Proxy

incC :: Natural -> ClientM ()
getC :: ClientM Natural

incC :<|> getC = client api

semantics :: ClientEnv -> Command resp -> IO resp
semantics env cmd = fmap (either (error . show) id) $ flip runClientM env $
  case cmd of
    Inc n -> incC n
    Get   -> getC

data SomeCommand where
  SomeCommand :: (Eq resp, Show resp) => Command resp -> SomeCommand

deriving instance Show SomeCommand

instance Arbitrary SomeCommand where
  arbitrary = oneof [ SomeCommand <$> (Inc <$> fmap (fromInteger . abs) arbitrary)
                    , pure (SomeCommand Get)
                    ]

setup :: IO ThreadId
setup = forkIO main

clean :: IO ()
clean = writeIORef counter initCounter

makeClientEnv :: IO ClientEnv
makeClientEnv = do
  mgr <- newManager defaultManagerSettings
  burl <- parseBaseUrl "http://localhost:8081"
  return (mkClientEnv mgr burl)

withWebServer :: IO () -> IO ()
withWebServer io = bracket setup killThread (const io)
```

This property tests that the using the client bindings to interact with the fake
server matches the specification.

```haskell
prop_counter :: [SomeCommand] -> Property
prop_counter cmds = monadicIO $ do
  liftIO clean
  env <- liftIO makeClientEnv
  monitor (collect (length cmds))
  go env initCounter cmds
  where
    go :: ClientEnv -> Counter -> [SomeCommand] -> PropertyM IO Bool
    go _env _c []                       = return True
    go env   c (SomeCommand cmd : cmds) = do
      resp <- run (semantics env cmd)
      let resp' = spec c cmd
      if resp == resp'
      then go env (transition c cmd) cmds
      else do
        monitor (counterexample (show resp ++ " /= \n" ++ show resp'))
        return False
```

Here's a fake server based on the state machine specification. We can give this
server to another team to test against using the client bindings. Because we
have tested this server against the specification, we can once the real server
is implemented rerun the same tests against it and if those tests pass we can be
fairly sure that given the other team the real server in place for the fake one
will not break anything for them.

```haskell
server :: Server API
server = incH :<|> getH
  where
    incH :: Natural -> Handler ()
    incH = liftIO . incM

    getH :: Handler Natural
    getH = liftIO getM

main :: IO ()
main = Warp.run 8081 app

app :: Application
app = serve api server
```

To sum up, here are the main points I want to put across:

  * Determinicity and speed;
  * Easier fault injection;
  * Modular development.

I've thought about how to combine these ideas for quite some time. I appear to
have opened an
[issue](https://github.com/advancedtelematic/quickcheck-state-machine/issues/177)
regarding this topic back in October 2017. Inside the issue are a couple of links to
my sources of inspiration. The closest to what I've presented here's the work of
[Edsko de Vries](https://github.com/edsko) et al which is documented in the
following [blog post](http://www.well-typed.com/blog/2019/01/qsm-in-depth/).

[Robert Danitz](https://github.com/rdanitz) has also written a REPL-like
interface for state machines written in the style described in Edsko's blog
post, which inspired me to think of more generic interfaces than a REPL and
which eventially made me realise that using a global variable is probably the
simplest solution.

### Towards simulation testing with contract tests

Now that we understand how to derive test double fakes from state machine
models, and how to test that the real implementation matches the state machine
specification, let's explore how those ideas are related to something called
_simulation testing_.

I first heard about simulation testing from Will Wilson's Strange Loop
[talk](https://www.youtube.com/watch?v=4fFDFbi3toc) called _Testing Distributed
Systems w/ Deterministic Simulation_, where he describes how they use it to
[test](https://apple.github.io/foundationdb/testing.html)
[FoundationDB](https://en.wikipedia.org/wiki/FoundationDB).

If you haven't heard of simulation testing before, then I'd highly recommend
watching the talk, as I shall only give a brief summary of the technique here.

The idea is similar to that of test double fakes, we often depend on components
that are non-deterministic and this can cause flaky tests, but simulation takes
this idea to its extream. To do simulation testing we must remove all sources of
non-determinism in our programs and replace them with deterministic fakes. These
determinstic fakes can be parametrised a random seed in order to achieve
non-determinism in a controlled and reproducable way.

Some common sources of non-determinism include:

  1. Filesystem interaction
  2. Pseudorandom number generation
  3. Timestamps
  4. Networking

The problem with simulation testing is similar to that with test doubles -- how
do we know that the simulation is correct? Will Wilson says in the talk that
this problem kept his CTO awake at nights.

But as we have seen above, we now know how to test our filesystem fakes against
the real filesystem interactions that they represent. Pseudorandom number
generation and timestamp fakes don't have any tricky assumptions about them as
far as I know, so testing them against the real components shouldn't be so
important. Networking however could be more a complicated. I think I can see how
to simulate a network, but how to test the simulation against a real network is
still not clear to me.

By combining simulation testing with fault injection we can test properties such
as linearisability, but much faster than the black-box approach of Jepsen. In
fact the FoundationDB people pushed this idea so far that Kyle
"[aphyr](https://github.com/aphyr)" Kingsbury (the main guy behind Jepsen)
[said](https://twitter.com/aphyr/status/405017101804396546) it wasn't worth
writing Jepsen tests for FoundationDB:

> "haven't tested foundation in part because their testing appears to be waaaay
> more rigorous than mine."

The main idea here is that because it's a simulation we can run the tests faster
than in the real world. This is important because of faster feedback loop when
developing, but also because it lets us find bugs faster than our users do as we
can simulate, say, one month's worth of user interaction (in rare corner case
scenarios due to fault injection) in a couple of seconds.

Apart from Will Wilson's talk that I've already mentioned, I have also found
[Tyler Neely's](https://github.com/spacejam)
[talk](https://www.youtube.com/watch?v=hMJEPWcSD8w) (PDF
[slides](https://github.com/spacejam/slides/blob/master/reliable_infrastructure_in_rust.pdf))
and
[writings](https://medium.com/@tylerneely/reliable-systems-series-model-based-property-testing-e89a433b360)
on simulation testing useful. Occationally he also makes shorter insightful
comments on
[reddit](https://old.reddit.com/r/rust/comments/cwfgqv/announcing_actixraft_raft_distributed_consensus/eyb42tg/),
[lobsters](https://lobste.rs/s/igiolo/learning_build_distributed_systems#c_nlpl7r)
and [twitter](https://twitter.com/sadisticsystems/status/1120702046624264192).
  
Will Wilson has also given a another
[talk](https://www.youtube.com/watch?v=fFSPwJFXVlw) in which he mentions he has
started a company whose business idea seems to be to deploy simulation testing
solutions to other companies with hard testing problems. This seems to suggest,
at least to me, that there's still a lot of work to be done on this topic and
that it can potentially be very useful when developing certain types of
applications, e.g. distributed systems.

If you are interested in this kind of testing, or know others that are or even
perhaps have done something similar, do feel free to get in touch. I would be
particularly interested in collaborating with people, from different programming
language communities, on making the ideas of model- and simulation-based testing
more accessible and effective.
