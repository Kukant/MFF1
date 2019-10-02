//TASK Build a processor for user scripts that will be provided in plain text at runtime.
//On the input the processor receives a list of urls and a user script that performs user-defined filtering of these urls.
//The processor should return urls of the websites that were returned by the user script.
//The user scripts must be allowed to use custom commands - 'download', 'unless', siteTalksAboutGroovy', 'remember' and 'rememberedSites'.
//The semantics of these commands can be deduced from the 'test user script input', defined at the bottom of this assignment.
//Use the tricks we learnt about scripting, passing parameters to GroovyShell through binding, properties,
//closure delegates, the 'object.with(Closure)' method, etc.

List<String> filterSitesByUserScript(String userScript, List<String> sites) {
    //Filtering function. Needs to be implemented
    
    def unlessX = { boolean a, Closure c ->
        if (!a) c.run()
    }
    
    def binding = new Binding()
    def downloadX = { new URL(it).text }
    def siteTalksAboutGroovyX = {it.contains("groovy")}
    def rememberX = { rememberedSites += it }
    rememberX.resolveStrategy = Closure.DELEGATE_ONLY
    rememberX.delegate = binding
    
    binding.with {
        rememberedSites = []
        allSites = sites
        unless = unlessX
        download = downloadX
        siteTalksAboutGroovy = siteTalksAboutGroovyX
        remember = rememberX
    }

    GroovyShell shell = new GroovyShell(binding)
    shell.evaluate(userScript)
}

//************* Do not modify after this point!

//A test user script input.
String userInput = '''
    for(site in allSites) {
        def content = download site
        unless (siteTalksAboutGroovy(content)) {
            remember site
        }
    }
    return rememberedSites
'''

//Calling the filtering method on a list of sites.
sites = ["http://groovy.cz", "http://gpars.org", "http://groovy-lang.org/", "http://infoq.com", "http://oracle.com", "http://ibm.com"]
def result = filterSitesByUserScript(userInput, sites)
result.each {
    println 'No groovy mention at ' + it
}
assert result.size()>0 && result.size() <= sites.size
println 'ok'
